import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import numpy as np
import pandas as pd
from glob import glob
import cv2
import argparse, os
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm

import model.util as util


def extract_faces(args):
	tasks = []

	for video in tqdm(args.videos, desc='Registering tasks', leave=False):
		uid = os.path.basename(video).split('.')[0]

		for segment in ['01', '02', '03']:
			file_name = f'{uid}_{segment}'

			frames_path = f'{args.asd_path}/tracklets/{file_name}'
			os.makedirs(frames_path, exist_ok=True)

			track = read_track(f'{args.asd_path}/predictions/{file_name}.csv')
			if track is None: continue

			for _, track_row in track.iterrows():
				frame_path = build_frame_path(frames_path, track_row)

				if not os.path.exists(frame_path):
					tasks.append((frame_path, track_row, video))

	# process_map(crop_and_align_frame, tasks, max_workers=args.n_threads, chunksize=4, desc=f'Cropping faces with {args.asd_detector}')
	with tqdm(desc=f'Cropping faces with {args.asd_detector}') as progress_bar:
		process_map(crop_frame, tasks, max_workers=args.n_threads, chunksize=4)


def read_track(path):
	if not os.path.exists(path):
		print(f'Track: {path} not found.')
		return None

	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label','entity_id', 'spkid']
	track = pd.read_csv(path, header=None, names=colnames, na_values=[''], keep_default_na=False)

	track = track[track['label'].str.contains('SPEAKING_AUDIBLE') & track['spkid'].str.contains('spk')]

	return track


def crop_frame(data):
	frame_path, track_row, video = data
	timestamp = track_row['frame_timestamp']

	cam = cv2.VideoCapture(video)
	cam.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1e3)

	_, frame = cam.read()

	if frame is None: return

	h = np.size(frame, 0)
	w = np.size(frame, 1)

	x1 = int(track_row['entity_box_x1'] * w)
	y1 = int(track_row['entity_box_y1'] * h)
	x2 = int(track_row['entity_box_x2'] * w)
	y2 = int(track_row['entity_box_y2'] * h)

	face_crop = frame[y1: y2, x1: x2, :]
	if face_crop.size == 0: return

	face_crop = cv2.resize(face_crop, (224, 224))

	cv2.imwrite(frame_path, face_crop)


def build_frame_path(frames_path, row):
	u_frame_id = f'{row['entity_id']}:{row['frame_timestamp']:.2f}'
	binary_label = int(row['label'] == 'SPEAKING_AUDIBLE')
	spkid = row['spkid']

	return f'{frames_path}/{u_frame_id}:{binary_label}:{spkid}.jpg'


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Location of dataset to process')
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Name of folder containing tracks to use in processing')
	parser.add_argument('--n_threads',  	type=int, default=10, 							help='Number of threads for preprocessing')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path = util.get_path('videos_path', data_type=args.data_type)
	args.videos = sorted(glob(f'{args.videos_path}/*.*'))

	args.asd_path = util.get_path('asd_path', data_type=args.data_type, asd_detector=args.asd_detector)
	args.tracklets_path = f'{args.asd_path}/tracklets'
	args.predictions_path = f'{args.asd_path}/predictions'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	extract_faces(args)


if __name__ == '__main__':
	args = initialize_arguments()
	extract_faces(args)
