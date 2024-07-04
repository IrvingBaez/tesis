import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import argparse, cv2, os
from model import util
from glob import glob
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from model.third_party.RetinaFace.face_aligner import face_aligner


def align_faces(args):
	aligner = face_aligner()
	tasks = []

	for video_id in tqdm(args.video_ids, desc='Registering tasks', leave=False):
		frame_paths = glob(f'{args.tracklets_path}/{video_id}/*.*')

		target_folder = f'{args.aligned_tracklets_path}/{video_id}'
		os.makedirs(target_folder, exist_ok=True)

		for frame_path in frame_paths:
			frame_name = frame_path.split('/')[-1]
			target_path = f'{target_folder}/{frame_name}'

			if not os.path.isfile(target_path):
				tasks.append((frame_path, target_path, aligner))

	process_map(align_frame, tasks, max_workers=args.n_threads, chunksize=10)


def align_frame(data):
	frame_path, target_path, aligner = data

	frame = cv2.imread(frame_path)
	aligned_frame = aligner.align_face(frame)

	if aligned_frame is not None:
		frame = aligned_frame

	cv2.imwrite(target_path, frame)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for face alignment")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Location of dataset to process')
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Name of folder containing tracks to use in processing')
	parser.add_argument('--n_threads',  	type=int, default=6, 							help='Number of threads for preprocessing')

	args = util.argparse_helper(parser, **kwargs)

	args.asd_path = util.get_path('asd_path', data_type=args.data_type, asd_detector=args.asd_detector)
	args.tracklets_path = f'{args.asd_path}/tracklets'
	args.aligned_tracklets_path = f'{args.asd_path}/aligned_tracklets'

	folder_paths = sorted(glob(f'{args.tracklets_path}/*'))
	args.video_ids = [path.split('/')[-1] for path in folder_paths]

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	align_faces(args)


if __name__ == '__main__':
	args = initialize_arguments()
	align_faces(args)
