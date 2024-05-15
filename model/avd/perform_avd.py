import argparse
import model.util as util
from model.avd.score_avd import main as score_avd

from model.third_party.avr_net.predict import main as avr_net


def perform_avd(args):
	if args.avd_detector == 'avr_net':
		arguments = {
			'videos_path':	args.videos_path,
			'waves_path': 	args.waves_path,
			'labs_path': 		args.labs_path,
			'frames_path': 	args.frames_path,
			'tracks_path': 	args.tracks_path,
			'sys_path': 		args.sys_path
		}
		avr_net(**arguments)

	score_avd(data_type=args.data_type, denoiser=args.denoiser, vad_detector=args.vad_detector, asd_detector=args.asd_detector, avd_detector=args.avd_detector)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Data type to process')
	parser.add_argument('--denoiser', 		type=str, default="dihard18", 		help='Model used to denoise video sound')
	parser.add_argument('--vad_detector', type=str, default="ground_truth", help='Voice activity detector to find off-screen speakers')
	parser.add_argument('--asd_detector', type=str, default="ground_truth", help='Active speacker detection used for face cropping')
	parser.add_argument('--avd_detector', type=str, default="avr_net", 			help='Model to use for audio visual diarozation')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path	= util.get_path('videos_path', data_type=args.data_type)
	args.waves_path 	= util.get_path('waves_path', data_type=args.data_type, denoiser=args.denoiser)
	args.labs_path 		= util.get_path('vad_path', data_type=args.data_type, denoiser=args.denoiser, vad_detector=args.vad_detector) + '/predictions'

	asd_path 					= util.get_path('asd_path', data_type=args.data_type, asd_detector=args.asd_detector)
	args.frames_path 	= f'{asd_path}/aligned_tracklets'
	args.tracks_path 	= f'{asd_path}/predictions'

	args.ref_path = util.get_path('avd_path', data_type=args.data_type, denoiser='dihard18', vad_detector='ground_truth', asd_detector='ground_truth', avd_detector= 'ground_truth')
	args.sys_path = util.get_path('avd_path', data_type=args.data_type, denoiser=args.denoiser, vad_detector=args.vad_detector, asd_detector=args.asd_detector, avd_detector= args.avd_detector)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	perform_avd(args)


if __name__ == '__main__':
	args = initialize_arguments()
	perform_avd(args)
