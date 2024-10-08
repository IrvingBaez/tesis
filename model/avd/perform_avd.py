import argparse, os
import model.util as util
from datetime import datetime
from model.avd.score_avd import main as score_avd

from model.third_party.avr_net.predict import main as avr_net

def perform_avd(args):
	arguments = {
		'data_type':		args.data_type,
		'video_ids':		','.join(args.video_ids),
		'videos_path':	args.videos_path,
		'waves_path': 	args.waves_path,
		'labs_path': 		args.labs_path,
		'rttms_path': 	args.rttms_path,
		'frames_path': 	args.frames_path,
		'tracks_path': 	args.tracks_path,
		'sys_path': 		args.sys_path
	}

	if args.avd_detector == 'avr_net':
		avr_net(**arguments)
	if args.avd_detector == 'avar_net':
		raise NotImplementedError("AVAR_NET is not ready")
		# avar_net(**arguments)

	score_avd_validation(args)


def score_avd_validation(args):
	score_avd(data_type=args.data_type, avd_detector=args.avd_detector)

	score_path = util.get_path('avd_path', avd_detector = args.avd_detector) + '/scores.out'
	with open(score_path, 'r') as file:
		scores = file.read()

	save_dir = 'model/third_party/avr_net/results'
	os.makedirs(save_dir, exist_ok=True)

	timestamp = datetime.now().strftime('%Y_%b_%d_%H:%M:%S')
	with open(f'{save_dir}/{timestamp}.out', 'w') as file:
		file.write(f'data_type:    {args.data_type}\n')
		file.write(f'denoiser:     {args.denoiser}\n')
		file.write(f'vad_detector: {args.vad_detector}\n')
		file.write(f'asd_detector: {args.asd_detector}\n')
		file.write(f'avd_detector: {args.avd_detector}\n')
		file.write(f'aligned:      {args.aligned}\n\n')
		file.write(scores)

	print(scores.split('\n')[-1])

def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Data type to process')
	parser.add_argument('--denoiser', 		type=str, default="dihard18", 		help='Model used to denoise video sound')
	parser.add_argument('--vad_detector', type=str, default="ground_truth", help='Voice activity detector to find off-screen speakers')
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Active speacker detection used for face cropping')
	parser.add_argument('--avd_detector', type=str, default="avr_net", 			help='Model to use for audio visual diarozation')
	parser.add_argument('--aligned', 			action='store_true', 							help='Used aligned frame crops')

	args = util.argparse_helper(parser, **kwargs)

	with open(f'dataset/split/{args.data_type}.list', 'r') as file:
		args.video_ids = file.read().split('\n')

	args.videos_path	= util.get_path('videos_path')
	args.waves_path 	= util.get_path('waves_path', denoiser=args.denoiser)
	args.labs_path 		= util.get_path('vad_path', vad_detector=args.vad_detector) + '/predictions'
	args.rttms_path 	= util.get_path('avd_path', avd_detector='ground_truth') + '/predictions'

	asd_path 					= util.get_path('asd_path', asd_detector=args.asd_detector)
	args.frames_path 	= f'{asd_path}/aligned_tracklets' if args.aligned else f'{asd_path}/tracklets'
	args.tracks_path 	= f'{asd_path}/predictions'

	args.sys_path = util.get_path('avd_path', avd_detector= args.avd_detector)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	perform_avd(args)


if __name__ == '__main__':
	args = initialize_arguments()
	perform_avd(args)
