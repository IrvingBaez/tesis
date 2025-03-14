import argparse, os
from shutil import rmtree
from glob import glob
from tqdm import tqdm

from model.third_party.light_asd import predict as light_asd_predict
from model.third_party.TalkNet import predict as talknet_predict
import model.util as util
import model.asd.score_asd as score_asd
import model.asd.visualize_asd as visualize_asd


def perform_asd(args):
	os.makedirs(args.save_csv_path, 					exist_ok = True)
	os.makedirs(args.save_score_path, 				exist_ok = True)
	os.makedirs(args.save_visualization_path, exist_ok = True)

	speaker_detector = get_detector(args)

	with open('dataset/split/train.list', 'r') as file:
		train_list = file.read().split('\n')

	progress_bar = tqdm(args.videos, desc=f'Performing ASD with {args.asd_detector}')
	for video in progress_bar:
		video_name = video.split('/')[-1].split('.')[0]
		progress_bar.set_description(f'Performing ASD with {args.asd_detector} on {video_name}')

		for segment in tqdm(range(1, 4), leave=False):
			gt_path							=	f'{args.load_gt_path}/{video_name}_0{segment}.csv'
			pred_path						=	f'{args.save_csv_path}/{video_name}_0{segment}.csv'
			score_path					=	f'{args.save_score_path}/{video_name}_0{segment}.out'
			visualization_path	=	f'{args.save_visualization_path}/{video_name}_0{segment}.avi'

			# There's never a reason to perform ASD on train set.
			if f'{video_name}_0{segment}' in train_list: continue

			if os.path.exists(pred_path):
				continue

			start = 600 + 300 * segment
			speaker_detector.main(video_folder=args.videos_path, video_name=video_name, csv_path=pred_path, num_loader_treads=args.workers, start=start, duration=300)

			if os.path.exists(video.split('.')[0]):
				rmtree(video.split('.')[0])

			if os.path.exists(gt_path):
				score_asd.main(gt_path=gt_path, pred_path=pred_path, save_path=score_path)

			if args.visualize:
				audio_path = f'{args.waves_path}/{video_name}.wav'
				visualize_asd.main(video_path=video, csv_path=pred_path, gt_path=gt_path, output_path=visualization_path, audio_path=audio_path)

	get_total_asd_score(args)


def get_detector(args):
	if args.asd_detector == 'light_asd':
		return light_asd_predict
	if args.asd_detector == 'talk_net':
		return talknet_predict


def get_total_asd_score(args):
	score_files = glob(f'{args.save_score_path}/*.*')

	mAPs = []
	F1s = []

	for score_file in score_files:
		with open(score_file, 'r') as file:
			lines = file.readlines()

			mAPs.append(float(lines[-2].split('\t')[-1]))
			F1s.append(float(lines[-1].split('\t')[-1]))

	avg_mAP = sum(mAPs) / len(mAPs)
	avg_F1 = sum(F1s) / len(F1s)

	print(f'Average mAP:\t{avg_mAP:0.4f}')
	print(f'Average F1:\t{avg_F1:0.4f}')

	with open(f'{args.save_path}/score.out', 'w') as file:
		file.write(f'Average mAP:\t{avg_mAP:0.4f}')
		file.write(f'Average F1:\t{avg_F1:0.4f}')


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for Active Speaker Detection")

	parser.add_argument('--data_type',		type=str, default="test", 			help='Type of dataset to process')
	parser.add_argument('--asd_detector',	type=str, default='light_asd',	help='Active Speaker Detector')
	parser.add_argument('--workers',			type=int, default=10,						help='Max number of workers for ASD')
	parser.add_argument('--verbose', 		action='store_true', help='Print progress and process')
	parser.add_argument('--visualize', 	action='store_true', help='Make video to visualize AVD predictions vs ground truth')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path 	= util.get_path('videos_path')
	args.waves_path		= util.get_path('waves_path', denoiser='original')
	args.load_path 		= util.get_path('asd_path', asd_detector='ground_truth')
	args.save_path 		= util.get_path('asd_path',	asd_detector=args.asd_detector)

	args.load_gt_path 						= f'{args.load_path}/predictions'
	args.save_csv_path 						= f'{args.save_path}/predictions'
	args.save_score_path 					= f'{args.save_path}/scores'
	args.save_visualization_path 	= f'{args.save_path}/visualization'

	args.videos = sorted(glob(f'{args.videos_path}/*.*'))

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	perform_asd(args)


if __name__ == '__main__':
	args = initialize_arguments()
	perform_asd(args)
