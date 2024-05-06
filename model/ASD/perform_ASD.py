import glob, argparse, os
from tqdm import tqdm
from shutil import rmtree
from model.util import argparse_helper

from model.third_party.Light_ASD import predict as light_asd_predict
import ground_truth_detector
import score_ASD
import visualize_ASD


def get_detector(args):
	if args.system == 'ground_truth':
		return ground_truth_detector
	if args.system == 'light_asd':
		return light_asd_predict
	if args.system == 'talk_net':
		# TODO: Implement TalkNet
		return


# TODO: Generate lab files.
def perform_asd(args):
	os.makedirs(args.csv_path, exist_ok = True)
	os.makedirs(args.score_path, exist_ok = True)
	os.makedirs(args.visualization_path, exist_ok = True)

	speaker_detector = get_detector(args)

	for video in tqdm(args.videos, desc=f'Performing ASD with {args.system}'):
		video_name = video.split('/')[-1].split('.')[0]
		gt_path = f'{args.gt_path}/{video_name}.csv'
		pred_path=f'{args.csv_path}/{video_name}.csv'
		save_path=f'{args.score_path}/{video_name}.out'
		visualization_path=f'{args.visualization_path}/{video_name}.avi'

		speaker_detector.main(video_folder=args.videos_path, video_name=video_name, csv_path=args.csv_path, verbose=args.verbose)
		if os.path.exists(video.split('.')[0]):
			rmtree(video.split('.')[0])

		if os.path.exists(gt_path):
			score_ASD.main(gt_path=gt_path, pred_path=pred_path, verbose=args.verbose, save_path=save_path)

		if args.visualize:
			visualize_ASD.main(video_path=video, csv_path=pred_path, gt_path=gt_path, output_path=visualization_path)

	get_total_asd_score(args)


def get_total_asd_score(args):
	scores_path = f'{args.data_path}/asd/f{args.system}/scores'
	score_files = glob(f'{scores_path}/*.*')

	mAPs = []
	F1s = []

	for score_file in score_files:
		with open(score_file, 'r') as file:
			lines = file.readlines()

			mAPs.append(float(lines[-2].split('\t')[-1]))
			F1s.append(float(lines[-1].split('\t')[-1]))

	avg_mAP = sum(mAPs) / len(mAPs)
	avg_F1 = sum(F1s) / len(F1s)

	with open(f'{scores_path}/__total_score.out', 'w') as file:
		file.write(f'Average mAP:\t{avg_mAP:0.4f}')
		file.write(f'Average F1:\t{avg_F1:0.4f}')


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for Active Speaker Detection")

	parser.add_argument('--data_path',	type=str, default="dataset/val", help='Location of dataset to process')
	parser.add_argument('--system',  		type=str, default='light_asd', help='System to use for Active Speaker Detection', choices=['ground_truth', 'light_asd', 'talk_net'])
	parser.add_argument('--verbose', 		action='store_true', help='Print progress and process')
	parser.add_argument('--visualize', 	action='store_true', help='Make video to visualize AVD predictions vs ground truth')

	args = argparse_helper(parser, **kwargs)

	args.videos_path = f'{args.data_path}/videos'
	args.gt_path = f'{args.data_path}/asd/ground_truth/predictions'

	args.csv_path = f'{args.data_path}/asd/{args.system}/predictions'
	args.score_path = f'{args.data_path}/asd/{args.system}/scores'
	args.visualization_path = f'{args.data_path}/asd/{args.system}/visualization'

	args.videos = glob.glob(f'{args.videos_path}/*.*')

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	perform_asd(args)


if __name__ == '__main__':
	args = initialize_arguments()
	perform_asd(args)
