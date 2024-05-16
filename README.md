# TODO: Write this file

# Title #

## Dependencies ##

Start from building the environment

conda create -n TalkNet python=3.12.2 anaconda
conda activate TalkNet
pip install -r requirement.txt

Start from the existing environment

pip install -r requirement.txt

## Dataset ##

To download AVA-AVD dataset, run:

`python3 dataset/scripts/download.py`

generate requirements.txt:
`conda list --explicit > conda_file.txt`
`pip freeze | grep -v ' @ file://' > requirements.txt`

Available to download:
- videos
- tracks
- rttms
- labs

For demo, run:
`python3 model/ASD.py -dp demo --verbose --visualize`