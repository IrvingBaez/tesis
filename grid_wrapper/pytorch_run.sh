# Activate your virtual environment
env_name='ava-exp'

source /export/b03/carlosc/miniconda3/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate ${env_name}

export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=$(free-gpu -n 1) python  "$@"
