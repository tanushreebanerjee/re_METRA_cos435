#!/bin/bash
#SBATCH --job-name=pixel-final    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # choose a100 or v100 on adroit

module purge
HOME=/scratch/network/tb19
module load anaconda3/2024.2
module load cudnn/cuda-11.5/8.3.2
conda activate metra
source ~/.bashrc

python tests/main.py --run_group Debug --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0
