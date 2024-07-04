import argparse
import model.third_party.dscore.score as dscore
import model.util as util


def score(args):
	dscore.main(ref_rttm_scpf=args.ref_path, sys_rttm_scpf=args.sys_path, output_path=args.output_path, collar=0.25)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for scoring Audio Visual Diarization")

	parser.add_argument('--data_type',		type=str, default="val", 					help='Dataset type to score')
	parser.add_argument('--avd_detector',	type=str, default="avr_net", 			help='Detector used to generate rttms')

	args = util.argparse_helper(parser, **kwargs)

	args.sys_path  		=	util.get_path('avd_path', avd_detector = args.avd_detector) + f'/{args.data_type}.out'
	args.output_path 	= util.get_path('avd_path', avd_detector = args.avd_detector) + '/scores.out'
	args.ref_path  		= util.get_path('avd_path', avd_detector = 'ground_truth') 		+ f'/{args.data_type}.out'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	score(args)


if __name__ == '__main__':
	args = initialize_arguments()
	score(args)
