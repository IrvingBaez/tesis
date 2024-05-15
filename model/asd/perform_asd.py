import argparse, os, torch
from shutil import rmtree
from glob import glob
from tqdm import tqdm

from model.third_party.light_asd import predict as light_asd_predict
import model.util as util
import model.asd.score_asd as score_asd
import model.asd.visualize_asd as visualize_asd


def perform_asd(args):
	os.makedirs(args.save_csv_path, 					exist_ok = True)
	os.makedirs(args.save_score_path, 				exist_ok = True)
	os.makedirs(args.save_visualization_path, exist_ok = True)

	speaker_detector = get_detector(args)

	for video in tqdm(args.videos, desc=f'Performing ASD with {args.asd_detector}'):
		video_name					= video.split('/')[-1].split('.')[0]
		gt_path							=	f'{args.load_gt_path}/{video_name}.csv'
		pred_path						=	f'{args.save_csv_path}/{video_name}.csv'
		score_path					=	f'{args.save_score_path}/{video_name}.out'
		visualization_path	=	f'{args.save_visualization_path}/{video_name}.avi'

		speaker_detector.main(video_folder=args.videos_path, video_name=video_name, csv_path=args.save_csv_path) #, verbose=args.verbose)

		if os.path.exists(video.split('.')[0]):
			rmtree(video.split('.')[0])

		if os.path.exists(gt_path):
			score_asd.main(gt_path=gt_path, pred_path=pred_path, save_path=score_path) # , verbose=args.verbose)

		if args.visualize:
			audio_path = f'{args.waves_path}/{video_name}.wav'
			visualize_asd.main(video_path=video, csv_path=pred_path, gt_path=gt_path, output_path=visualization_path, audio_path=audio_path)

		torch.cuda.empty_cache()

	get_total_asd_score(args)


def get_detector(args):
	if args.asd_detector == 'light_asd':
		return light_asd_predict
	if args.asd_detector == 'talk_net':
		# TODO: Implement TalkNet
		return


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
	parser.add_argument('--verbose', 		action='store_true', help='Print progress and process')
	parser.add_argument('--visualize', 	action='store_true', help='Make video to visualize AVD predictions vs ground truth')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path 	= util.get_path('videos_path',	data_type=args.data_type)
	args.waves_path		= util.get_path('waves_path',		data_type=args.data_type, denoiser='original')
	args.load_path 		= util.get_path('asd_path',			data_type=args.data_type)
	args.save_path 		= util.get_path('asd_path',			data_type=args.data_type, asd_detector=args.asd_detector)

	args.load_gt_path 						= f'{args.load_path}/predictions'
	args.save_csv_path 						= f'{args.save_path}/predictions'
	args.save_score_path 					= f'{args.save_path}/scores'
	args.save_visualization_path 	= f'{args.save_path}/visualization'

	args.videos = glob(f'{args.videos_path}/*.*')

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	perform_asd(args)


if __name__ == '__main__':
	args = initialize_arguments()
	perform_asd(args)
