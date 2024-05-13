import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import argparse, os, subprocess, cv2
import pandas as pd
import numpy as np
import noisereduce
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
from tqdm.contrib.concurrent import process_map

import model.util as util
from model.third_party.RetinaFace.face_aligner import face_aligner


def extract_waves(args):
	os.makedirs(args.original_waves_path, exist_ok=True)
	os.makedirs(args.denoised_waves_path, exist_ok=True)

	tasks = []

	for video in args.videos:
		uid = os.path.basename(video).split('.')[0]
		original_path = f'{args.original_waves_path}/{uid}.wav'
		denoised_path = f'{args.denoised_waves_path}/{uid}.wav'

		tasks.append((video, original_path, denoised_path, args.denoiser))

	process_map(extract_wave, tasks, max_workers=6, chunksize=1, desc='Extracting waves')


def extract_wave(data):
		video, original_path, denoised_path, denoiser = data

		command = [
				'ffmpeg',
				'-y',									# Overwrite output files without asking
				'-i', video,					# Input file
				'-qscale:a', '0',			# Audio quality scale (VBR), 0 is best quality
				'-ac', '1',						# Set audio channels to mono
				'-vn',								# No video (audio only)
				'-threads', '6',			# Number of threads to use
				'-ar', '16000',				# Audio sampling rate in Hz
				original_path,				# Output file path and name
				'-loglevel', 'panic'	# Only log critical errors
		]

		subprocess.call(command)

		rate, data = wavfile.read(original_path)

		if denoiser == 'noisereduce':
			wave = noisereduce.reduce_noise(y=data, sr=rate)
			wavfile.write(denoised_path, rate, wave)


def crop_align_faces(args):
	aligner = face_aligner()
	tasks = []

	for video in args.videos:
		uid = os.path.basename(video).split('.')[0]

		frames_path = f'{args.asd_path}/aligned_tracklets/{uid}'
		os.makedirs(frames_path, exist_ok=True)

		track = read_track(f'{args.asd_path}/predictions/{uid}.csv')
		if track is None: continue

		for _, track_row in track.iterrows():
			tasks.append((frames_path, track_row, video, aligner))

	process_map(crop_and_align_frame, tasks, max_workers=args.n_threads, chunksize=1, desc='Cropping and aligning faces')


def read_track(path):
	if not os.path.exists(path):
		print(f'Track: {path} not found.')
		return None

	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label','entity_id', 'spkid']
	track = pd.read_csv(path, header=None, names=colnames)
	track = track[track['spkid'].str.contains('spk')]

	return track


def crop_and_align_frame(data):
	frames_path, track_row, video, aligner = data

	frame_path = build_frame_path(frames_path, track_row)
	if os.path.exists(frame_path):
		print(f'Frame already exists: {frame_path}')
		return

	cam = cv2.VideoCapture(video)
	cam.set(cv2.CAP_PROP_POS_MSEC, track_row['frame_timestamp'] * 1000)
	success, frame = cam.read()

	if not success:
		print(f'Video not found: {video}')
		return

	heigt = np.size(frame, 0)
	width = np.size(frame, 1)

	x1 = int(track_row['entity_box_x1'] * width)
	y1 = int(track_row['entity_box_y1'] * heigt)
	x2 = int(track_row['entity_box_x2'] * width)
	y2 = int(track_row['entity_box_y2'] * heigt)
	face_crop = frame[y1:y2, x1:x2, :]
	face_crop = cv2.resize(face_crop, (224, 224))

	aligned_face_crop = aligner.align_face(face_crop)

	if aligned_face_crop is not None:
		face_crop = aligned_face_crop

	cv2.imwrite(frame_path, face_crop)


def build_frame_path(frames_path, row):
	u_frame_id = f'{row['entity_id']}:{row['frame_timestamp']:.2f}'
	binary_label = int(row['label'] == 'SPEAKING_AUDIBLE')
	spkid = row['spkid']

	return f'{frames_path}/{u_frame_id}:{binary_label}:{spkid}.jpg'


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Location of dataset to process')
	parser.add_argument('--denoiser',			type=str, default="noisereduce", 	help='Location of dataset to process')
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Name of folder containing tracks to use in processing')
	parser.add_argument('--n_threads',  	type=int, default=2, 							help='Number of threads for preprocessing')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path = util.get_path('videos_path', data_type=args.data_type)
	args.videos = glob(f'{args.videos_path}/*.*')

	args.original_waves_path = util.get_path('waves_path', data_type=args.data_type, denoiser='original')
	args.denoised_waves_path = util.get_path('waves_path', data_type=args.data_type, denoiser=args.denoiser)
	args.asd_path = util.get_path('asd_path', data_type=args.data_type, asd_detector=args.asd_detector)
	args.tracklets_path = f'{args.asd_path}/aligned_tracklets'
	args.predictions_path = f'{args.asd_path}/predictions'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)

	extract_waves(args)
	crop_align_faces(args)


if __name__ == '__main__':
	args = initialize_arguments()

	extract_waves(args)
	crop_align_faces(args)
