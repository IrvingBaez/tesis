import argparse
import model.util as util
import os
from datetime import datetime
from model.avd.score_avd import main as score_avd
from pathlib import Path

from model.third_party.avr_net.predict import main as avr_net


def perform_avd(args):
	arguments = {
		'data_type':			'val',
		'video_ids':			','.join(args.video_ids),
		'sys_path': 			args.sys_path,
		'checkpoint':			args.checkpoint,
		'aligned':				args.aligned,
		'max_frames':			args.max_frames,
		'db_video_mode': 	args.db_video_mode
	}

	avr_net(**arguments)

	score_avd_validation(args)


def score_avd_validation(args):
	score_path = f'{args.sys_path}/val_scores.out'
	score_avd(data_type='val', sys_path=f'{args.sys_path}/val.out', output_path=score_path)

	with open(score_path, 'r') as file:
		scores = file.read()

	timestamp = datetime.now().strftime('%Y_%b_%d_%H:%M:%S')
	with open(f'{args.save_dir}/{timestamp}.out', 'w') as file:
		file.write(f'data_type:    val\n')
		file.write(f'aligned:      {args.aligned}\n\n')
		file.write(scores)

	print(scores.split('\n')[-1])


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--checkpoint',			type=str,							help='Checkpoint to evaluate')
	parser.add_argument('--max_frames',			type=int,							help='How many frames to use in self-attention')
	parser.add_argument('--aligned', 				action='store_true',	help='Used aligned frame crops')
	parser.add_argument('--db_video_mode', 	type=str, 						help='Selection mode for video frames in the dataset')

	args = util.argparse_helper(parser, **kwargs)

	with open(f'dataset/split/val.list', 'r') as file:
		args.video_ids = file.read().split('\n')

	args.save_dir = f'model/third_party/avr_net/results/{args.checkpoint.split('/')[5]}'
	args.sys_path = args.save_dir

	os.makedirs(args.save_dir, exist_ok=True)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	perform_avd(args)


if __name__ == '__main__':
	args = initialize_arguments()
	perform_avd(args)
