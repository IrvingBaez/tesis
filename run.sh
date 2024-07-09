env_name='tesis'

eval "$(conda shell.bash hook)"
conda activate ${env_name}

export TF_CPP_MIN_LOG_LEVEL=2
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$PWD

python3 model/experiments.py