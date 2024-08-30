#!/bin/bash
#
#SBATCH --job-name=avr_net_experiments #job name
#SBATCH --nodes=1 #number of nodes requested
#SBATCH --gpus=4 #number of gpus requested
#SBATCH --partition=gpu #queue
#SBATCH --mail-user="irvingbaezc@gmail.com" #email for reporting
#SBATCH --mail-type=ALL #report types

set -eu

source /etc/profile.d/modules.sh
module purge
module load conda
module load ffmpeg
module load cuda/12.1

source /home/ibaez6/.bashrc
conda activate avr-net
export TF_CPP_MIN_LOG_LEVEL=2
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$PWD

python3 model/experiments.py
/bin/hostname # print used node
