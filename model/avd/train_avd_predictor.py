import argparse, os
import model.util as util
from glob import glob
from datetime import datetime
from model.avd.score_avd import main as score_avd

from model.third_party.avr_net.train import main as train_avr_net
from model.third_party.avr_net.predict import main as predict


# TODO: Training is always done with the same, data. Is this file, really necesary?
def train_avd_predictor(args):
	if args.avd_detector == 'avr_net':
		arguments = {
			'video_ids':		','.join(args.video_ids),
			'videos_path':	args.videos_path,
			'waves_path': 	args.waves_path,
			'labs_path': 		args.labs_path,
			'frames_path': 	args.frames_path,
			'rttms_path': 	args.rttms_path,
			'sys_path': 		args.sys_path,
			'epochs':				args.epochs
		}
		train_avr_net(**arguments)

		arguments['data_type'] = args.val_data_type
		arguments['video_ids'] = ','.join(args.val_video_ids)
		arguments['weights_path'] = sorted(glob('model/third_party/avr_net/checkpoints/*.ckpt'))[-1]
		del arguments['rttms_path']
		predict(**arguments)

	args.weights = arguments['weights_path']
	score = score_avd_validation(args)
	print(f'DER: {score}')
	return score


def score_avd_validation(args):
	score_avd(data_type=args.val_data_type, avd_detector=args.avd_detector)

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
		file.write(f'aligned:      {args.aligned}\n')
		file.write(f'weights:      {args.weights}\n\n')
		file.write(scores)

	summary = scores.split('\n')[-1]
	return float(summary.split()[3])


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--denoiser', 		type=str, default="dihard18", 		help='Model used to denoise video sound')
	parser.add_argument('--vad_detector', type=str, default="ground_truth", help='Voice activity detector to find off-screen speakers')
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Active speacker detection used for face cropping')
	parser.add_argument('--avd_detector', type=str, default="avr_net", 			help='Model to use for audio visual diarozation')
	parser.add_argument('--aligned', 			action='store_true', 							help='Used aligned frame crops')
	parser.add_argument('--epochs', 			type=int, default=100,						help='Used aligned frame crops')

	args = util.argparse_helper(parser, **kwargs)

	args.data_type = 'train'
	args.val_data_type = 'val'

	with open(f'dataset/split/{args.data_type}.list', 'r') as file:
		args.video_ids = file.read().split('\n')

	with open(f'dataset/split/val.list', 'r') as file:
		args.val_video_ids = file.read().split('\n')

	args.videos_path	= util.get_path('videos_path')
	args.waves_path 	= util.get_path('waves_path', denoiser=args.denoiser)
	args.labs_path 		= util.get_path('vad_path', vad_detector=args.vad_detector) + '/predictions'

	asd_path 					= util.get_path('asd_path', asd_detector=args.asd_detector)
	args.frames_path 	= f'{asd_path}/aligned_tracklets' if args.aligned else f'{asd_path}/tracklets'
	args.tracks_path 	= f'{asd_path}/predictions'

	args.rttms_path = util.get_path('avd_path', avd_detector='ground_truth') + '/predictions'
	args.sys_path 	= util.get_path('avd_path', avd_detector= args.avd_detector)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	return train_avd_predictor(args)


if __name__ == '__main__':
	args = initialize_arguments()
	train_avd_predictor(args)
