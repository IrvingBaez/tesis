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
export NCCL_P2P_DISABLE='1'

program="model/experiments.py"
if [ "$#" -gt 0 ]; then
    program="$@"
fi

python3 $program