#!/bin/bash
#
#SBATCH --job-name=avr_net_experiments 			# Job name
#SBATCH --nodes=1 													# Number of nodes requested
#SBATCH --mem=50G														# RAM requested
#SBATCH --gpus=4 														# Number of gpus requested
#SBATCH --partition=gpu 										# Queue
#SBATCH --exclude=c04,c18,c19								# Avoid these nodes
#SBATCH --mail-user="irvingbaezc@gmail.com"	# Email for reporting
#SBATCH --mail-type=ALL 										# Report types

set -eu

source /etc/profile.d/modules.sh
module purge
module load conda
module load ffmpeg
module load cuda/12.1

source /home/ibaez6/.bashrc
conda activate avr-net
/bin/hostname # print used node

export TF_CPP_MIN_LOG_LEVEL=2
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Clean up .pyc and __pycache__ before running
# find . -name "*.pyc" -o -name "__pycache__" | xargs rm -rf

python3 model/experiments.py