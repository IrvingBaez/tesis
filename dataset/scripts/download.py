import gdown, zipfile, os, subprocess
from tqdm.auto import tqdm


def main():
	save_dir = 'dataset/videos'
	os.makedirs(save_dir, exist_ok=True)

	with open('dataset/split/video.list', 'r') as f:
		videos = f.readlines()

	for video in tqdm(videos):
		if os.path.isfile(f'{save_dir}/{video.strip()}'): continue

		cmd = f'wget -P {save_dir} https://s3.amazonaws.com/ava-dataset/trainval/{video.strip()}'
		subprocess.call(cmd, shell=True)

	gdown.download('1MZIfZRLug1t2o3I8tReC8eTu9V76mtHi')


if __name__ == '__main__':
	main()
