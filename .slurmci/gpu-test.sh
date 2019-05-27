#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-23-38 # has libcuda issue

set -euo pipefail

# to avoid race conditions
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot_gpu"
export PATH="${PATH}:${HOME}/julia-1.2-gpu/bin"

set -x #echo on

module load cmake/3.10.2 openmpi/3.1.2 cuda/9.1

julia --color=no --project=env/gpu test/runtests.jl
