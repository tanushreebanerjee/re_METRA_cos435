import torch
from torch.nn import functional as F


def _clip_actions(algo, actions):
    epsilon = 1e-6
    action_space = algo._env_spec.action_space

    # Convert action space bounds to tensors once and reuse
    lower_bound = torch.tensor(action_space.low, dtype=torch.float32, device=algo.device) + epsilon
    upper_bound = torch.tensor(action_space.high, dtype=torch.float32, device=algo.device) - epsilon

    # Use torch.clamp for efficient clipping within bounds
    with torch.no_grad():
        clipped_actions = torch.clamp(actions, min=lower_bound, max=upper_bound)

    return clipped_actions


def update_loss_qf(algo, tensors, v, obs, actions, next_obs, dones, rewards, policy):
    """ Update the Q-function losses for a batch of data and update the tensors dictionary with relevant metrics. """
    # Compute the exponentiated value of log_alpha without gradient calculations
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    # Predict current Q-values for both critics
    q1_pred = algo.qf1(obs, actions).flatten()
    q2_pred = algo.qf2(obs, actions).flatten()

    # Sample and log probabilities of next actions using the policy's action distribution
    new_next_actions, new_next_action_log_probs = sample_next_actions(policy, next_obs, algo)

    # Calculate target Q-values using the target critics and adjusted for entropy term
    target_q_values = calculate_target_q_values(algo, next_obs, new_next_actions, alpha, new_next_action_log_probs)

    # Compute the final target Q-values considering done flags
    q_target = compute_final_target_q_values(rewards, target_q_values, dones, algo.discount)

    # Calculate MSE loss for both Q-functions and weigh them by 0.5 as per the critic loss weight
    loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
    loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

    # Update the tensors dictionary with computed losses and other metrics
    tensors.update({
        'QTargetsMean': q_target.mean(),
        'QTdErrsMean': ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
        'LossQf1': loss_qf1,
        'LossQf2': loss_qf2,
    })

def sample_next_actions(policy, next_obs, algo):
    """ Sample next actions based on policy and optionally clip them. """
    next_action_dists, *_ = policy(next_obs)
    if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
        new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
        new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
    else:
        new_next_actions = next_action_dists.rsample()
        new_next_actions = _clip_actions(algo, new_next_actions)
        new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)
    return new_next_actions, new_next_action_log_probs

def calculate_target_q_values(algo, next_obs, new_next_actions, alpha, new_next_action_log_probs):
    """ Calculate target Q-values adjusted for entropy and using target networks. """
    target_q_values = torch.min(
        algo.target_qf1(next_obs, new_next_actions).flatten(),
        algo.target_qf2(next_obs, new_next_actions).flatten(),
    )
    target_q_values -= alpha * new_next_action_log_probs
    return target_q_values

def compute_final_target_q_values(rewards, target_q_values, dones, discount):
    """ Finalize the target Q-values, adjusting for episode completion. """
    return rewards + (1. - dones) * discount * target_q_values


def update_loss_sacp(algo, tensors, v, obs, policy):
    """ Update the Soft Actor-Critic policy (SACP) loss and log related metrics. """
    # Calculate alpha value without gradient computation
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    # Sample new actions using the provided policy and calculate their log probabilities
    new_actions, new_action_log_probs = sample_and_log_probs(policy, obs, algo)

    # Compute the minimum Q-values from the dual Q-networks for the new actions
    min_q_values = compute_min_q_values(algo, obs, new_actions)

    # Calculate the SACP loss based on the difference between scaled log probs and Q-values
    loss_sacp = compute_sacp_loss(alpha, new_action_log_probs, min_q_values)

    # Update the tensors dictionary with the new calculated metrics
    tensors.update({
        'SacpNewActionLogProbMean': new_action_log_probs.mean(),
        'LossSacp': loss_sacp,
    })

    # Store new action log probabilities in v for potential future use
    v.update({'new_action_log_probs': new_action_log_probs})

def sample_and_log_probs(policy, obs, algo):
    """ Sample actions from the policy and calculate their log probabilities, handling action clipping if necessary. """
    action_dists, *_ = policy(obs)
    if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
    else:
        new_actions = action_dists.rsample()
        new_actions = _clip_actions(algo, new_actions)
        new_action_log_probs = action_dists.log_prob(new_actions)
    return new_actions, new_action_log_probs

def compute_min_q_values(algo, obs, actions):
    """ Compute the minimum Q-values from the two Q-functions for given actions. """
    return torch.min(
        algo.qf1(obs, actions).flatten(),
        algo.qf2(obs, actions).flatten(),
    )

def compute_sacp_loss(alpha, log_probs, min_q_values):
    """ Calculate the Soft Actor-Critic Policy loss. """
    return (alpha * log_probs - min_q_values).mean()


def update_loss_alpha(algo, tensors, v):
    """ Calculate and update the loss for the alpha parameter based on the target entropy and new action log probabilities. """
    log_alpha = algo.log_alpha.param
    target_entropy = algo._target_entropy
    detached_log_probs = v['new_action_log_probs'].detach()

    # Calculate the loss for alpha, which controls the entropy regularization
    loss_alpha = (-log_alpha * (detached_log_probs + target_entropy)).mean()

    # Update tensors dictionary with the current alpha value and its loss
    tensors.update({
        'Alpha': log_alpha.exp(),  # Exponentiated to get the actual alpha value
        'LossAlpha': loss_alpha,
    })


def update_targets(algo):
    """ Soft update the parameters of the target Q-functions using the Polyak averaging method. """
    # Retrieve target and main Q-function pairs
    target_qfs = [algo.target_qf1, algo.target_qf2]
    qfs = [algo.qf1, algo.qf2]

    # Update each target Q-function's parameters
    for target_qf, qf in zip(target_qfs, qfs):
        soft_update(target_qf, qf, algo.tau)

def soft_update(target, source, tau):
    """ Perform soft update of target network parameters from source network parameters using the tau parameter. """
    for target_param, param in zip(target.parameters(), source.parameters()):
        updated_value = (1 - tau) * target_param.data + tau * param.data
        target_param.data.copy_(updated_value)