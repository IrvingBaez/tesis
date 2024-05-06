#!/bin/bash
#
#SBATCH --job-name=avr_net									#job name
#SBATCH --nodes=1														#number of nodes requested
#SBATCH --gpus=4														#number of gpus requested
#SBATCH --partition=gpu											#queue
#SBATCH --mail-user="irvingbaezc@gmail.com”	#email for reporting
#SBATCH --mail-type=ALL 										#report types

module purge
module load conda
conda create --name ava-exp --file /export/c03/carlosc/ibaez/tesis/conda_file.txt
conda activate ava-exp
pip3 install -r requirements.txt
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH='/export/c03/carlosc/ibaez/tesis'
python3 /export/c03/carlosc/ibaez/tesis/model/third_party/avr_net/predict.py