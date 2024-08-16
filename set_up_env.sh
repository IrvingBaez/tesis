#!/bin/bash
set -eu

env_name='tesis'
source ~/miniconda3/etc/profile.d/conda.sh

if conda info --envs | grep -q ${env_name}
then
  echo "Environment ${env_name} already exists"
else
  echo "Creating environment ${env_name}"
	conda create --name ${env_name} --file conda_file.txt

	eval "$(conda shell.bash hook)"
	conda activate ${env_name}

	pip3 install -r requirements.txt
fi
