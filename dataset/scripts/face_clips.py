import cv2
import os
import pickle
import subprocess
from glob import glob
from tqdm.auto import tqdm
from moviepy.editor import ImageSequenceClip


def get_db_items():
	# TODO: code here
	pass


def video_frame_rates():
	frame_rates = {}
	for video in glob('dataset/videos/*.*'):
		cam = cv2.VideoCapture(video)
		fps = cam.get(cv2.CAP_PROP_FPS)

		video_name = video.split('/')[-1].split('.')[0]
		frame_rates[video_name] = fps

	return frame_rates


def make_clips():
	save_path = 'dataset/face_clips'
	frame_rates = video_frame_rates()

	with open("dataset/scripts/db_items.pckl", "rb") as file:
		data = pickle.load(file)

	for clip in tqdm(data, desc='Creating face clips'):
		sound_path, start, duration, offset, frames, video_id = clip
		if not frames: continue

		os.makedirs(f'{save_path}/{video_id}', exist_ok=True)

		video_name = video_id.rsplit('_', 1)[0]
		frame_rate = frame_rates[video_name]

		spkid = frames[0].split(':')[-1].split('.')[0]
		temp_clip_name = f'{save_path}/{video_id}/{video_name}:{start}:{spkid}_mute.avi'
		clip_name = f'{save_path}/{video_id}/{video_name}:{start}:{spkid}.avi'

		start -= offset
		video_clip = ImageSequenceClip(frames, fps=frame_rate)
		video_clip.write_videofile(temp_clip_name, codec="ffv1", ffmpeg_params=["-crf", "0"], verbose=False, logger=None)

		cmd = f'ffmpeg -y -ss {start} -t {video_clip.duration} -i {sound_path} -r {frame_rate} -i {temp_clip_name} -filter:a aresample=async=1 -c:a flac -c:v copy {clip_name}'
		subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

		os.remove(temp_clip_name)


if __name__ == '__main__':
	make_clips()