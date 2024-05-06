env_name='ava-exp'

source /export/b03/carlosc/miniconda3/etc/profile.d/conda.sh

if conda info --envs | grep -q ${env_name}
then
  echo "Environment ${env_name} already exists"
else
  echo "Creating environment ${env_name}"
	conda create --name ${env_name} --file conda_file.txt
fi

eval "$(conda shell.bash hook)"
conda activate ${env_name}

pip3 install -r requirements.txt
