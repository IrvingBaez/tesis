import argparse
import model.third_party.dscore.score as dscore
import model.util as util


def score(args):
	dscore.main(ref_rttm_scpf=args.ref_path, sys_rttm_scpf=args.sys_path, output_path=args.output_path)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for scoring Audio Visual Diarization")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Dataset type to score')
	parser.add_argument('--denoiser',			type=str, default="dihard18", 		help='Detector used to generate rttms')
	parser.add_argument('--vad_detector',	type=str, default="ground_truth", help='Detector used to generate rttms')
	parser.add_argument('--asd_detector',	type=str, default="ground_truth", help='Detector used to generate rttms')
	parser.add_argument('--avd_detector',	type=str, default="avr_net", 			help='Detector used to generate rttms')

	args = util.argparse_helper(parser, **kwargs)

	args.sys_path  = util.get_path(
		'avd_path',
		data_type		 = args.data_type,
		denoiser		 = args.denoiser,
		vad_detector = args.vad_detector,
		asd_detector = args.asd_detector,
		avd_detector = args.avd_detector)

	args.ref_path  = util.get_path(
		'avd_path',
		data_type		 = args.data_type,
		denoiser		 = 'dihard18',
		vad_detector = 'ground_truth',
		asd_detector = 'ground_truth',
		avd_detector = 'ground_truth') + '/rttms.out'

	args.output_path = args.sys_path + '/scores.out'
	args.sys_path += '/rttms.out'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	score(args)


if __name__ == '__main__':
	args = initialize_arguments()
	score(args)
