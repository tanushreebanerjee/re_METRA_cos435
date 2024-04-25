# things to make METRA work

## Activate the environment on adroit.
```Bash
HOME=/scratch/network/tb19 # need to make the default into the scratch location for more GBs
module load anaconda3/2024.2
```
Environment is setup according to the metra instructions https://github.com/seohongpark/METRA?tab=readme-ov-file#installation. 
```Bash
conda activate metra # after following installation instructions, activate the created environment
```

## Running the first test script:
```Bash
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2
```

---
Problem: Mujoco error:
```
You appear to be missing MuJoCo.  We expected to find the file here: /scratch/network/tb19/.mujoco/mujoco210

This package only provides python bindings, the library must be installed separately.
```
Solution: Follow these instructions to install mujoco and extract it to the required location (~/.mujoco/mujoco210): https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

---
Problem: Environment variables need to be configured in .bashrc

Solution: Follow the instructions as suggested to add the following to the .bashrc
```Bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/network/tb19/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
---
Problem: Missing `patchelf` on system. Used in `mujoco_py`. We don't have admin priveleges on adroit, so we can't simply download using a package manager. 

Solution: Install and compile `patchelf` from source code (https://github.com/NixOS/patchelf)
- Download tar of `patchelf` release, upload using `scp` to the adroit cluster, unpack, and then run installation commands:
```Bash
./bootstrap.sh
./configure --prefix=/scratch/network/tb19/local
make
make check
make install
```
Then add `export PATH=$PATH:/scratch/network/tb19/local/bin` to `~/.bashrc` and run `source ~/.bashrc`

---
Problem: No CUDA GPU available on adroit (when not using SLURM scheduling)

Quick Fix: add `--use_gpu 0` to the test command.

Solution: Schedule the job using SLURM and allocate necessary GPUs

---
Problem: Cudnn version mismatch, symbol lookup error.

Solution: Manually install the correct cuda-toolkit
```Bash
conda install nvidia/label/cuda-11.5.0::cuda-toolkit
```
And make sure to include the correct modules in adroit.
```Bash
module purge
HOME=/scratch/network/tb19 # if using the scratch network for more gbs
module load anaconda3/2024.2 # for conda
module load cudnn/cuda-11.5/8.3.2 # cudnn version compatible for cudatoolkit 11.5
conda activate metra 
source ~/.bashrc
```

# Reimplementing METRA Algorithm

## General Improvements

**Modularization:** Key computations and logic segments were extracted into separate, well-defined functions. This not only simplifies each function but also enhances readability and maintainability.

**Improved Readability:** By using descriptive naming, adding detailed comments, and separating concerns, the functions are more understandable and easier to maintain.

**Efficiency and Performance:** Leveraged built-in PyTorch functionalities and optimized tensor operations to improve performance and reduce the risk of bugs.

## Specific Function Enhancements

- Rewrote `./iod/metra.py`
- Rewrote `./iod/sac_utils.py`
- Note: Did not change overall code flow to ensure input output consistency.

## TODOS

- Add code to `./main.py` for reproducing downstream tasks (Fig 6 and 9)
