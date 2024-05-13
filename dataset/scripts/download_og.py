import os, subprocess
from glob import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip


def download_annotations():
	cmd = f'gdown --id 18kjJJbebBg7e8umI6HoGE4_tI3OWufzA'
	subprocess.call(cmd, shell=True)
	cmd = f'tar -xvf annotations.tar.gz -C dataset/'
	subprocess.call(cmd, shell=True)
	os.remove('annotations.tar.gz')

	for file_name in glob('dataset/rttms/*.*'):
		with open(file_name, 'r') as file:
			new_text = file.read().replace('_c_', '_')

		with open(file_name, 'w') as file:
			file.write(new_text)


def download_videos(data_type):
	with open(f'dataset/scripts/{data_type}.list', 'r') as f:
		videos = f.readlines()

	videos_path = f'dataset/{data_type}/videos'
	csvs_path 	= f'dataset/{data_type}/asd/ground_truth/predictions'
	rttms_path 	= f'dataset/{data_type}/avd/dihard18/ground_truth/ground_truth/ground_truth/predictions'
	labs_path 	= f'dataset/{data_type}/vad/ground_truth*predictions'

	os.makedirs(videos_path, 	exist_ok=True)
	os.makedirs(csvs_path, 		exist_ok=True)
	os.makedirs(rttms_path, 	exist_ok=True)
	os.makedirs(labs_path, 		exist_ok=True)

	for video in tqdm(videos):
		video = video.strip()
		video_path = f'dataset/{data_type}/videos/{video}'
		if os.path.isfile(video_path): continue

		cmd = f'wget -P {video_path} https://s3.amazonaws.com/ava-dataset/trainval/{video}'
		subprocess.call(cmd, shell=True)

		# tracks
		os.rename(f'dataset/tracks/{video}-activespeaker.csv', f'{csvs_path}/{video}.csv')

		# rttms
		os.rename(f'dataset/rttms/{video}_c_01.rttm', f'{rttms_path}/{video}_01.rttm')
		os.rename(f'dataset/rttms/{video}_c_02.rttm', f'{rttms_path}/{video}_02.rttm')
		os.rename(f'dataset/rttms/{video}_c_03.rttm', f'{rttms_path}/{video}_03.rttm')

		# labs
		os.rename(f'dataset/labs/{video}_c_01.lab', f'{labs_path}/{video}_01.lab')
		os.rename(f'dataset/labs/{video}_c_02.lab', f'{labs_path}/{video}_02.lab')
		os.rename(f'dataset/labs/{video}_c_03.lab', f'{labs_path}/{video}_03.lab')




def crop_video(start, end, name, new_name):
	clip = VideoFileClip(name).subclip(start, end)
	clip.to_videofile(new_name, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')


if __name__ == '__main__':
	download_annotations()

	for data_type in ['test', 'val', 'train']:
		download_videos(data_type)
