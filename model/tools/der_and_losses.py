from datetime import datetime
from glob import glob
import torch
from tqdm.auto import tqdm
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

from model import util
from model.third_party.avr_net.predict import main as predict
from model.avd.train_avd_predictor import score_avd_validation as score

def save_scores(scores, path):
	with open(path, 'wb') as file:
		pickle.dump(scores, file)


def load_scores(path):
	try:
		with open(path, 'rb') as file:
			return pickle.load(file)
	except:
		return {}


def prediction_args(aligned:bool=True):
	arguments = {
		'video_ids': '2XeFK-DTSZk_01,2XeFK-DTSZk_02,2XeFK-DTSZk_03,55Ihr6uVIDA_01,55Ihr6uVIDA_02,55Ihr6uVIDA_03,914yZXz-iRs_01,914yZXz-iRs_02,914yZXz-iRs_03,QCLQYnt3aMo_01,QCLQYnt3aMo_02,QCLQYnt3aMo_03,fD6VkIRlIRI_01,fD6VkIRlIRI_02,fD6VkIRlIRI_03,iK4Y-JKRRAc_01,iK4Y-JKRRAc_02,iK4Y-JKRRAc_03,o4xQ-BEa3Ss_01,o4xQ-BEa3Ss_02,o4xQ-BEa3Ss_03,oD_wxyTHJ2I_01,oD_wxyTHJ2I_02,oD_wxyTHJ2I_03,rUYsoIIE37A_01,rUYsoIIE37A_02,rUYsoIIE37A_03,tt0t_a1EDCE_01,tt0t_a1EDCE_02,tt0t_a1EDCE_03,u1ltv6r14KQ_01,u1ltv6r14KQ_02,u1ltv6r14KQ_03,uPJPNPbWMFk_01,uPJPNPbWMFk_02,uPJPNPbWMFk_03,xmqSaQPzL1E_01,xmqSaQPzL1E_02,xmqSaQPzL1E_03,yMtGmGa8KZ0_01,yMtGmGa8KZ0_02,yMtGmGa8KZ0_03,8aMv-ZGD4ic_01,8aMv-ZGD4ic_02,8aMv-ZGD4ic_03,Hi8QeP_VPu0_01,Hi8QeP_VPu0_02,Hi8QeP_VPu0_03,PNZQ2UJfyQE_01,PNZQ2UJfyQE_02,PNZQ2UJfyQE_03,tjqCzVjojCo_01,tjqCzVjojCo_02,tjqCzVjojCo_03',
		'videos_path': 'dataset/videos',
		'waves_path': 'dataset/waves/dihard18',
		'labs_path': 'dataset/vad/ground_truth/predictions',
		'frames_path': 'dataset/asd/ground_truth/aligned_tracklets',
		'sys_path': 'model/third_party/avr_net/features',
		'data_type': 'val',
		'weights_path': None
	}

	return arguments


def score_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_type', 			type=str, default="val")
	parser.add_argument('--val_data_type', 	type=str, default="val")
	parser.add_argument('--vad_detector', 	type=str, default="ground_truth")
	parser.add_argument('--asd_detector', 	type=str, default="ground_truth")
	parser.add_argument('--avd_detector', 	type=str, default="avr_net")
	parser.add_argument('--denoiser', 			type=str, default="dihard18")
	parser.add_argument('--aligned', 				type=bool, default=True)
	parser.add_argument('--weights', 				type=str, default="")

	args = parser.parse_args([])
	return args


def collect_losses(args):
	epochs = []
	losses = []

	for path in tqdm(args.checkpoint_paths):
		checkpoint = torch.load(path)

		epoch_losses = []
		for value in checkpoint['losses']:
			if isinstance(value, int) or isinstance(value, float):
				epoch_losses.append(float(value))
				continue

			epoch_losses.append(value.cpu().detach().numpy()[0])

		avg_loss = sum(epoch_losses) / len(epoch_losses)
		epochs.append(checkpoint['epoch'])
		losses.append(avg_loss)

	return epochs, losses


def calculate_scores(args):
	all_scores = load_scores(args.scores_path)
	current_scores = []
	val_args = prediction_args()
	scoring_args = score_args()

	for index, path in enumerate(tqdm(args.checkpoint_paths)):
		if path in all_scores.keys():
			print(f'{index:03d}: Already calculated')
			current_scores.append(all_scores[path])
			continue

		if (index % args.test_every_n) != 0:
			print(f'{index:03d}: Skipping due to arg test_every_n')
			current_scores.append(None)
			continue

		if args.graph_ready:
			print(f'{index:03d}: Skipping due to arg graph_ready')
			current_scores.append(None)
			continue

		print(f'{index:03d}: Calculating...')

		val_args['weights_path'] = path
		scoring_args.weights = path

		predict(**val_args)
		all_scores[path] = score(scoring_args)

		save_scores(all_scores, args.scores_path)
		current_scores.append(all_scores[path])

	return current_scores


def der_and_losses(args):
	scores = calculate_scores(args)
	epochs, losses = collect_losses(args)

	epochs = np.array(epochs)
	scores = np.array(scores).astype(np.double)
	losses = np.array(losses).astype(np.double)

	scores_mask = np.isfinite(scores)
	losses_mask = np.isfinite(losses)

	plt.plot(epochs[scores_mask], scores[scores_mask], label = "DER score")
	plt.plot(epochs[losses_mask], losses[losses_mask], label = "losses")
	plt.xlabel('Epoch')
	plt.title("Training with Cross-Attention")
	plt.legend()

	timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
	plt.savefig(f'{args.outs_path}/trainig_cross-attention_{timestamp}.jpg')


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--ckpt_path',		type=str, default="model/third_party/avr_net/checkpoints_attention", help='Checkpoints to score')
	parser.add_argument('--outs_path', 		type=str, default="model/tools", help='Where to save the outputs of this script')
	parser.add_argument('--test_every_n', type=int, default=5, help='Obtain one score every n weights.')
	parser.add_argument('--graph_ready',	action='store_true', help='Graph ready scores without calculating new ones')
	args = util.argparse_helper(parser, **kwargs)

	args.scores_path = f'{args.outs_path}/losses_data.pckl'
	args.checkpoint_paths = sorted(glob(f'{args.ckpt_path}/*.*'))

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	der_and_losses(args)


if __name__ == '__main__':
	args = initialize_arguments()
	der_and_losses(args)
