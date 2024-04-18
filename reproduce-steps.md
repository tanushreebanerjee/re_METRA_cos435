# things to make METRA work

Activate the environment on adroit.
```Bash
HOME=/scratch/network/tb19 # need to make the default into the scratch location for more gigs
module load anaconda3/2024.2
conda activate metra
```
Environment is setup according to the instructions. Running the first test script:

Problem: Mujoco error:
```
You appear to be missing MuJoCo.  We expected to find the file here: /scratch/network/tb19/.mujoco/mujoco210

This package only provides python bindings, the library must be installed separately.
```
Solution: Follow these instructions to install mujoco and extract it to the required location (~/.mujoco/mujoco210): https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

Problem: Environment variables need to be configured in .bashrc
Solution: Follow the instructions as suggested to add the following to the .bashrc
```Bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/network/tb19/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Problem: Missing `patchelf` on system. Used in `mujoco_py`. We don't have admin priveleges on adroit, so we can't simply download using a package manager. 
Solution: Install and compile `patchelf` from source code (https://github.com/NixOS/patchelf)
- Download tar of `patchelf` release, upload using `scp` to the adroit cluster, then unpack.
```Bash
./bootstrap.sh
./configure --prefix=/scratch/network/tb19/local
make
make check
make install
```
Then add `export PATH=$PATH:/scratch/network/tb19/local/bin` to `~/.bashrc` and run `source ~/.bashrc`

Problem: No CUDA GPU available on adroit (when not using SLURM scheduling)
Quick Fix: add `--use-gpu 0` to the test command.
Solution: Schedule the job using SLURM and allocate necessary GPUs
