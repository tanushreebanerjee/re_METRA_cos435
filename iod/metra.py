import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians


class METRA(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,

            pixel_shape=None,

            **kwargs,
    ):
        super().__init__(**kwargs)

        # Move Q-functions and alpha to the appropriate device (assumed to be available in IOD or via self.device)
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        # Create target Q-functions by copying the initial ones
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        # Alpha parameter for entropy regularization
        self.log_alpha = log_alpha.to(self.device)

        # Dictionary to keep track of trainable parameters
        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        # Soft update rate
        self.tau = tau

        # Experience replay setup
        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        # Dual variables for optimization
        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        # Sampling and grouping controls
        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        # Scale reward by a fixed factor and calculate target entropy based on action space dimensions
        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        # Optional pixel-based input shape
        self.pixel_shape = pixel_shape

        # Ensure required optimization settings are defined
        assert self._trans_optimization_epochs is not None

    @property
    def policy(self):
        return {'option_policy': self.option_policy}

    def _get_concat_obs(self, observations, option):
        """ Concatenate observation and option vectors. """
        return get_torch_concat_obs(observations, option)

    def _get_random_options(self, batch_size):
        """ Helper to generate random options based on the option type. """
        if self.discrete:
            return np.eye(self.dim_option)[np.random.randint(0, self.dim_option, batch_size)]
        random_options = np.random.randn(batch_size, self.dim_option)
        return random_options / np.linalg.norm(random_options, axis=-1, keepdims=True) if self.unit_length else random_options

    def _get_train_trajectories_kwargs(self, runner):
        """ Prepare keyword arguments for training trajectories. """
        random_options = self._get_random_options(runner._train_args.batch_size)
        extras = self._generate_option_extras(random_options)
        return {'extras': extras, 'sampler_key': 'option_policy'}

    def _flatten_data(self, data):
        """ Flatten each data field and convert to a tensor. """
        return {key: torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device) for key, value in data.items()}

    def _update_replay_buffer(self, data):
        """ Update the replay buffer with new data paths. """
        if not self.replay_buffer:
            return

        # Prepare and add new paths to the replay buffer
        keys = data.keys()
        for action_index in range(len(data['actions'])):
            path = {key: data[key][action_index][..., None] if data[key][action_index].ndim == 1 else data[key][action_index]
                    for key in keys}
            self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        """ Sample transitions from the replay buffer and prepare them for processing. """
        # Sample transitions using the pre-defined batch size
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        
        # Transform samples into PyTorch tensors, adjusting shapes where necessary
        data = {key: torch.tensor(np.squeeze(value, axis=1) if value.shape[1] == 1 and 'option' not in key else value,
                                dtype=torch.float32, device=self.device)
                for key, value in samples.items()}
        
        return data

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        return tensors

    def _train_components(self, epoch_data):
        """ Conduct training steps for optimization epochs only if buffer conditions are met. """
        if self.replay_buffer and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}
        
        for _ in range(self._trans_optimization_epochs):
            tensors = self._train_epoch(epoch_data)

        return tensors

    def _train_epoch(self, epoch_data):
        """ Run a single training epoch using either minibatch tensors or sampled buffer data. """
        data = self._get_mini_tensors(epoch_data) if not self.replay_buffer else self._sample_replay_buffer()
        tensors = {}
        for func in [self._optimize_te, self._update_rewards, self._optimize_op]:
            func(tensors, data)
        return tensors

    def _optimize_te(self, tensors, v):
        """ Optimize trajectory encoding with possible regularization updates. """
        self._update_loss_te(tensors, v)
        self._perform_gradient_descent(tensors['LossTe'], 'traj_encoder')

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, v)
            self._perform_gradient_descent(tensors['LossDualLam'], 'dual_lam')

            if self.dual_dist == 's2_from_s':
                self._perform_gradient_descent(tensors['LossDp'], 'dist_predictor')

    def _optimize_op(self, tensors, v):
        """ Optimize option policies and Q-functions. """
        self._update_loss_qf(tensors, v)
        combined_loss = tensors['LossQf1'] + tensors['LossQf2']
        self._perform_gradient_descent(combined_loss, 'qf')

        self._update_loss_op(tensors, v)
        self._perform_gradient_descent(tensors['LossSacp'], 'option_policy')

        self._update_loss_alpha(tensors, v)
        self._perform_gradient_descent(tensors['LossAlpha'], 'log_alpha')

        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v):
        """ Compute rewards from current and next observations. """
        cur_z, next_z = self._compute_latent_variables(v)
        v.update({'cur_z': cur_z, 'next_z': next_z,})
        tensors.update(self._compute_rewards(v, cur_z, next_z))

    def _compute_latent_variables(self, v):
        cur_z = self.traj_encoder(v['obs']).mean
        next_z = self.traj_encoder(v['next_obs']).mean
        return cur_z, next_z

    def _compute_rewards(self, v, cur_z, next_z):
        target_z = next_z - cur_z
        rewards = self._calculate_inner_rewards(v, target_z) if self.inner else self._calculate_cross_entropy_rewards(v, target_z)
        v['rewards'] = rewards
        return {
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        }

    def _calculate_inner_rewards(self, v, target_z):
        if self.discrete:
            masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
            return (target_z * masks).sum(dim=1)
        return (target_z * v['options']).sum(dim=1)

    def _calculate_cross_entropy_rewards(self, v, target_z):
        logits = self.traj_encoder(v['next_obs']).mean if self.discrete else self.traj_encoder(v['next_obs']).log_prob(v['options'])
        return -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none') if self.discrete else logits

    def _perform_gradient_descent(self, loss, optimizer_key):
        """ Execute gradient descent on the specified loss using the designated optimizer. """
        self._gradient_descent(loss, optimizer_keys=[optimizer_key])

    def _update_loss_te(self, tensors, v):
        """ Update loss for trajectory encoder with potential dual regularizations. """
        # Update and retrieve rewards
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        # Prepare for dual distribution calculations if applicable
        obs, next_obs = v['obs'], v['next_obs']
        if self.dual_dist == 's2_from_s':
            tensors['LossDp'] = self._calculate_s2_from_s_loss(obs, next_obs)

        # Dual regularization calculations
        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            cst_dist = self._calculate_cst_dist(obs, next_obs, v)
            cst_penalty = self._calculate_cst_penalty(cst_dist, v['cur_z'], v['next_z'])
            te_obj = rewards + dual_lam.detach() * cst_penalty
            v['cst_penalty'] = cst_penalty
            tensors['DualCstPenalty'] = cst_penalty.mean()
        else:
            te_obj = rewards

        # Finalize the loss for trajectory encoder
        loss_te = -te_obj.mean()
        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _calculate_s2_from_s_loss(self, obs, next_obs):
        """ Calculate the s2_from_s distribution loss. """
        s2_dist = self.dist_predictor(obs)
        return -s2_dist.log_prob(next_obs - obs).mean()

    def _calculate_cst_dist(self, obs, next_obs, v):
        """ Calculate constraint distances based on different distribution metrics. """
        if self.dual_dist == 'l2':
            return torch.square(next_obs - obs).mean(dim=1)
        elif self.dual_dist == 'one':
            return torch.ones_like(obs[:, 0])
        elif self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            scaling_factor, normalized_scaling_factor = self._compute_scaling_factors(s2_dist)
            v.update({
                'ScalingFactor': scaling_factor.mean(dim=0),
                'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
            })
            s2_dist_mean = s2_dist.mean
            return torch.mean(torch.square((next_obs - obs) - s2_dist_mean) * normalized_scaling_factor, dim=1)
        else:
            raise NotImplementedError("Unknown dual_dist configuration.")

    def _compute_scaling_factors(self, s2_dist):
        """ Compute scaling factors for s2_from_s distance constraints. """
        s2_dist_std = s2_dist.stddev
        scaling_factor = 1. / s2_dist_std
        geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
        normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
        return scaling_factor, normalized_scaling_factor

    def _calculate_cst_penalty(self, cst_dist, phi_x, phi_y):
        """ Compute constraint penalties for the distance metrics. """
        cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
        return torch.clamp(cst_penalty, max=self.dual_slack)

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['next_obs']), v['next_options'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(v['obs']), v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _evaluate_policy(self, runner):
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)
