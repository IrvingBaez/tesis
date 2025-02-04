#!/bin/bash
set -eu

env_name='avr-net'
source ~/miniconda3/etc/profile.d/conda.sh

if conda info --envs | grep -q ${env_name}; then
  echo "Environment ${env_name} already exists, removing..."
	conda env remove --name ${env_name} -y
fi

echo "Creating environment ${env_name}"
conda create --name ${env_name} --file conda_file.txt -y

eval "$(conda shell.bash hook)"
conda activate ${env_name}

pip3 install -r requirements.txt
echo "Finished instalation of env ${env_name}"