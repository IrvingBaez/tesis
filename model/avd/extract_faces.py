import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import argparse, os, cv2
import pandas as pd
import numpy as np
from glob import glob
from tqdm.contrib.concurrent import process_map

import model.util as util
from model.third_party.RetinaFace.face_aligner import face_aligner


def extract_faces(args):
	aligner = face_aligner()
	tasks = []

	for video in args.videos:
		uid = os.path.basename(video).split('.')[0]

		frames_path = f'{args.asd_path}/aligned_tracklets/{uid}'
		os.makedirs(frames_path, exist_ok=True)

		track = read_track(f'{args.asd_path}/predictions/{uid}.csv')
		if track is None: continue

		for _, track_row in track.iterrows():
			frame_path = build_frame_path(frames_path, track_row)

			if not os.path.exists(frame_path):
				tasks.append((frame_path, track_row, video, aligner))

	process_map(crop_and_align_frame, tasks, max_workers=args.n_threads, chunksize=4, desc=f'Cropping faces with {args.asd_detector}')


def read_track(path):
	if not os.path.exists(path):
		print(f'Track: {path} not found.')
		return None

	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label','entity_id', 'spkid']
	track = pd.read_csv(path, header=None, names=colnames)

	track = track[track['label'].str.contains('SPEAKING_AUDIBLE')]

	return track


def crop_and_align_frame(data):
	frame_path, track_row, video, aligner = data

	cam = cv2.VideoCapture(video)
	if not cam.isOpened(): return

	cam.set(cv2.CAP_PROP_POS_MSEC, track_row['frame_timestamp'] * 1000)
	success, frame = cam.read()
	cam.release()

	if not success: return

	height, width = frame.shape[:2]

	x1 = int(track_row['entity_box_x1'] * width)
	y1 = int(track_row['entity_box_y1'] * height)
	x2 = int(track_row['entity_box_x2'] * width)
	y2 = int(track_row['entity_box_y2'] * height)
	face_crop = frame[y1:y2, x1:x2, :]

	if face_crop.size == 0: return

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
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Name of folder containing tracks to use in processing')
	parser.add_argument('--n_threads',  	type=int, default=2, 							help='Number of threads for preprocessing')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path = util.get_path('videos_path', data_type=args.data_type)
	args.videos = glob(f'{args.videos_path}/*.*')

	args.asd_path = util.get_path('asd_path', data_type=args.data_type, asd_detector=args.asd_detector)
	args.tracklets_path = f'{args.asd_path}/aligned_tracklets'
	args.predictions_path = f'{args.asd_path}/predictions'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	extract_faces(args)


if __name__ == '__main__':
	args = initialize_arguments()
	extract_faces(args)
