# TODO: Write this file

# Title #

## Dependencies ##


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