import glob, argparse, os, subprocess, cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

from model.ASD.util import cpu_parallel_process
from model.third_party.RetinaFace.face_aligner import face_aligner
# TODO: Try out this aligner: https://github.com/serengil/retinaface

def parse_arguments():
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('-dp', '--dataPath', dest='data_path',  type=str, default="dataset/train", help='Location of dataset to process')
	parser.add_argument('-tf', '--trackFolder', dest='track_folder',  type=str, default="tracks", help='Name of folder containing tracks to use in processing')
	parser.add_argument('-n', '--nThreads',  dest='n_threads', type=int, default=1, help='Number of threads for preprocessing')

	return parser.parse_args()


def extract_waves(arguments):
	videos, args = arguments
	os.makedirs(f'{args.data_path}/waves', exist_ok=True)

	for video in tqdm(videos, desc='Extracting waves'):
		uid = os.path.basename(video).split('.')[0]

		command = [
				'ffmpeg',
				'-y',                               	# Overwrite output files without asking
				'-i', video,                        	# Input file
				'-qscale:a', '0',                   	# Audio quality scale (VBR), 0 is best quality
				'-ac', '1',                         	# Set audio channels to mono
				'-vn',                              	# No video (audio only)
				'-threads', '6',                    	# Number of threads to use
				'-ar', '16000',                     	# Audio sampling rate in Hz
				f'{args.data_path}/waves/{uid}.wav',	# Output file path and name
				'-loglevel', 'panic'                	# Only log critical errors
		]
		subprocess.call(command)


def crop_align_face(arguments):
	videos, args = arguments
	aligner = face_aligner()

	for video in tqdm(videos, desc='Cropping and aligning faces'):
		uid = os.path.basename(video).split('.')[0]

		frames_path = f'{args.data_path}/aligned_tracklets/{uid}'
		os.makedirs(frames_path, exist_ok=True)

		track = read_track(f'{args.data_path}/{args.track_folder}/{uid}.csv')
		if track is None:
			continue

		cam = cv2.VideoCapture(video)
		for _, track_row in tqdm(track.iterrows(), desc=f'Processing track for {uid}', total=len(track)):
			cam.set(cv2.CAP_PROP_POS_MSEC, track_row['frame_timestamp'] * 1000)
			_, frame = cam.read()

			if frame is None:
				print(cam, track_row['frame_timestamp'], 'not found. Skipping.')
				continue

			binary_label, spkid, u_frame_id = row_data(track_row)
			aligned_face_crop = crop_and_align_frame(track_row, frame, aligner)

			cv2.imwrite(f'{frames_path}/{u_frame_id}:{binary_label}:{spkid}.jpg', aligned_face_crop)


def read_track(path):
	if not os.path.exists(path):
		print(f'Track: {path} not found.')
		return None

	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label','entity_id', 'spkid']
	track = pd.read_csv(path, header=None, names=colnames)
	track = track[track['spkid'].str.contains('spk')]

	return track


def row_data(row):
	return (
		int(row['label'] == 'SPEAKING_AUDIBLE'),
		row['spkid'],
		f'{row['entity_id']}:{row['frame_timestamp']:.2f}'
	)

def crop_and_align_frame(track_row, frame, aligner):
	heigt = np.size(frame, 0)
	width = np.size(frame, 1)

	x1 = int(track_row['entity_box_x1'] * width)
	y1 = int(track_row['entity_box_y1'] * heigt)
	x2 = int(track_row['entity_box_x2'] * width)
	y2 = int(track_row['entity_box_y2'] * heigt)
	face_crop = frame[y1:y2, x1:x2, :]
	face_crop = cv2.resize(face_crop, (224, 224))

	aligned_face_crop = aligner.align_face(face_crop)

	return face_crop if aligned_face_crop is None else aligned_face_crop


if __name__ == "__main__":
	args = parse_arguments()

	videos = glob.glob(f'{args.data_path}/videos/*.*')

	# cpu_parallel_process(extract_waves, videos, args.n_threads, args=args)
	cpu_parallel_process(crop_align_face, videos, args.n_threads, args)