import glob, argparse, os
from tqdm import tqdm
from shutil import rmtree

def parse_arguments():
	parser = argparse.ArgumentParser(description = "Arguments for Active Speaker Detection")

	parser.add_argument('-dp', '--dataPath', dest='data_path',  type=str, default="dataset/train", help='Location of dataset to process')
	parser.add_argument('-s', '--system',  dest='system', type=str, default='light_asd', help='System to use for Active Speaker Detection', choices=['ground_truth', 'light_asd'])
	parser.add_argument('--verbose', action='store_true', help='Print progress and process')
	parser.add_argument('--visualize', action='store_true', help='Make video to visualize AVD predictions vs ground truth')

	args = parser.parse_args()

	args.videos = glob.glob(args.data_path + '/videos/*.*')

	return args


# TODO: Implement TalkNet
def perform_asd(args, model):
	if model == 'ground_truth':
		return
	if model == 'light_asd':
		light_asd(args)


def light_asd(args):
	csvPath = f'{args.data_path}/ASD_predictions'
	scorePath = f'{args.data_path}/ASD_scores'
	gtPath = f'{args.data_path}/tracks'
	verbose_flag = '--verbose' if args.verbose else ''

	os.makedirs(csvPath, exist_ok = True)
	os.makedirs(scorePath, exist_ok = True)

	for video in tqdm(args.videos, desc=f'Performing ASD with light_asd', disable=not args.verbose):
		video_name = video.split('/')[-1].split('.')[0]
		ground_truth_exists = os.path.exists(f'{gtPath}/{video_name}.csv')

		os.system(f"python3 third_party/Light_ASD/predict.py {verbose_flag}  --videoFolder='{args.data_path}/videos' --videoName='{video_name}' --csvPath='{csvPath}'")
		rmtree(f'{args.data_path}/videos/{video_name}')

		if ground_truth_exists:
			os.system(f"python3 model/score_ASD.py -gt {gtPath}/{video_name}.csv -pr {csvPath}/{video_name}.csv --verbose > {scorePath}/{video_name}.out")

		if args.visualize:
			gt_argument = f" -gtp {gtPath}/{video_name}.csv" if ground_truth_exists else ''
			os.system(f"python3 model/visualize_ASD.py -vp {video} -cp {csvPath}/{video_name}.csv {gt_argument}")


if __name__ == "__main__":
	args = parse_arguments()

	perform_asd(args, args.system)
