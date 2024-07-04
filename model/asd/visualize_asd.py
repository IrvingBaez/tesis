import cv2, argparse, os, subprocess
import pandas as pd
from tqdm import tqdm
from math import isnan

from model.util import argparse_helper


def visualization(args):
	video = cv2.VideoCapture(args.video_path)

	fps 		= video.get(cv2.CAP_PROP_FPS)
	height 	= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width 	= int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

	track = quantize_csv(args.csv_path, height, width, fps)
	ground_truth = quantize_csv(args.gt_path, height, width, fps)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	output = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

	frame_count = 0
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	with tqdm(total=total_frames, desc='Rendering visualization', leave=False) as progress_bar:
		while True:
			ret, frame = video.read()

			if not ret:
				break

			# For the track
			rectangles = track.loc[track['frame_timestamp'] == frame_count]
			draw_rectangles(frame, rectangles, ground_truth=False)

			# For Ground Truth
			rectangles = ground_truth.loc[ground_truth['frame_timestamp'] == frame_count]
			draw_rectangles(frame, rectangles, ground_truth=True)

			output.write(frame)
			frame_count += 1
			progress_bar.update(1)

	video.release()
	output.release()
	# cv2.destroyAllWindows()

	with_sound_path = args.output_path.split('.')[0] + '_sound.avi'
	cmd = f'ffmpeg -y -i {args.audio_path} -r 30 -i {args.output_path}  -filter:a aresample=async=1 -c:a flac -c:v copy {with_sound_path}'
	subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def quantize_csv(csv_path, height, width, fps):
	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label', 'entity_id', 'spkid']

	if csv_path is None or not os.path.exists(csv_path):
		return pd.DataFrame(columns=colnames)

	track = pd.read_csv(csv_path, header=None, names=colnames)

	track = track.sort_values('frame_timestamp')

	track['entity_box_x1'] = track['entity_box_x1'].apply(lambda x: int(x * width))
	track['entity_box_y1'] = track['entity_box_y1'].apply(lambda x: int(x * height))
	track['entity_box_x2'] = track['entity_box_x2'].apply(lambda x: int(x * width))
	track['entity_box_y2'] = track['entity_box_y2'].apply(lambda x: int(x * height))

	track['frame_timestamp'] = track['frame_timestamp'].apply(lambda x: int(x * fps))

	return track


def draw_rectangles(frame, df_row, ground_truth):
	intensity = 127 if ground_truth else 255

	for _, rentangle in df_row.iterrows():
		point_1 = (rentangle['entity_box_x1'], rentangle['entity_box_y1'])
		point_2 = (rentangle['entity_box_x2'], rentangle['entity_box_y2'])
		color = (0, intensity, 0) if rentangle['label'] == 'SPEAKING_AUDIBLE' else (0, 0, intensity)
		thickness = 2

		cv2.rectangle(frame, point_1, point_2, color, thickness)
		# if isinstance(rentangle['entity_id'], str):
		# 	cv2.putText(frame, rentangle['entity_id'], (point_1[0], point_2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for ASD vizualization")

	parser.add_argument('--video_path', 	type=str, help='Location of video to process', 					required=True)
	parser.add_argument('--csv_path', 		type=str, help='Location of tracks csv to draw', 				required=True)
	parser.add_argument('--gt_path', 			type=str, help='Location of ground truth csv to draw', 	default=None)
	parser.add_argument('--output_path',	type=str, help='Path to write the output video', 				default='')
	parser.add_argument('--audio_path',		type=str, help='Path to write the output video', 				default='')

	args = argparse_helper(parser, **kwargs)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	visualization(args)


if __name__ == '__main__':
	args = initialize_arguments()
	visualization(args)
