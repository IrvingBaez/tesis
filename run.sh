#!/bin/bash
set -eu

env_name='tesis'

source ~/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate ${env_name}

export TF_CPP_MIN_LOG_LEVEL=3
export NO_ALBUMENTATIONS_UPDATE=1
export TF_USE_LEGACY_KERAS=1
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0

# For torch.nn.parallel.DistributedDataParallel
export MASTER_ADDR='localhost'
export MASTER_PORT='12355'
# export NCCL_P2P_DISABLE='1'
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

python3 model/experiments.py || spd-say 'Experiments failed'
spd-say 'Finished running experiments'