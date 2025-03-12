import torch, argparse
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import pytorch_lightning as pl

from model.third_party.avr_net.train_lightning import Lightning_Attention_AVRNet
from model.third_party.avr_net.tools.custom_collator import CustomCollator
from model.third_party.avr_net.tools.clustering_dataset import ClusteringDataset
from model.third_party.avr_net.tools.write_rttms import main as write_rttms
from model.util import argparse_helper, save_data, check_system_usage, show_similarities
from .feature_extraction import main as extract_features


def predict(args):
	similarities = compute_similarity(args)
	save_data(similarities, args.similarities_path, verbose=True, override=True)
	show_similarities('similarities_testing', similarities['similarities'])

	write_rttms(similarities_path=args.similarities_path, sys_path=args.sys_path, data_type=args.data_type)


def compute_similarity(args):
	similarities = {}

	args.train_features_path, args.val_features_path = extract_features(
		sys_path=args.sys_path,
		aligned=args.aligned,
		max_frames=args.max_frames,
		disable_pb=args.disable_pb
	)

	dataset, dataloader = load_data(args, mode='val', workers=2, batch_size=1024)
	model = load_model(args)
	trainer = pl.Trainer(logger=False)

	predictions = trainer.predict(model, dataloader)
	del trainer, model, dataloader
	
	for video_id in args.video_ids:
		similarities[video_id] = torch.diag_embed(torch.ones([dataset.utterance_counts[video_id]]))

	for batch in tqdm(predictions, desc='Organizing predictions'):
		for prediction in batch:
			video_id, index_a, index_b, score = prediction

			similarities[video_id][index_a, index_b] = score
			similarities[video_id][index_b, index_a] = score
			del prediction

	return {'similarities': similarities, 'starts': dataset.starts(), 'ends': dataset.ends()}


def load_model(args):
	model = Lightning_Attention_AVRNet.load_from_checkpoint(args.checkpoint)
	model.eval()

	return model


def load_data(args, mode, workers, batch_size=256):
	dataset = ClusteringDataset(args.val_features_path, args.disable_pb, video_proportion=1.0)
	sampler = SequentialSampler(dataset)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=workers,
		persistent_workers=True,
		sampler=sampler,
		pin_memory=True,
		drop_last=False,
		collate_fn=CustomCollator()
	)

	return dataset, dataloader


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "AVR_Net prediction")

	parser.add_argument('--data_type',		type=str,	help='Type of data being processed, test, val or train')
	parser.add_argument('--video_ids',		type=str,	help='Video ids separated by commas')
	parser.add_argument('--sys_path',			type=str,	help='Path to the folder where to save all the system outputs')
	parser.add_argument('--max_frames',		type=int,	help='How many frames to use in self-attention')
	parser.add_argument('--aligned', 			action='store_true',	help='Used aligned frame crops')
	parser.add_argument('--disable_pb', 	action='store_true',	help='If true, hides progress bars')

	# MODEL CONFIGURATION
	parser.add_argument('--checkpoint', 	type=str,	help='Path of checkpoint to load and evaluate')

	args = argparse_helper(parser, **kwargs)

	args.video_ids = args.video_ids.split(',')
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	args.similarities_path = f'{args.sys_path}/similarity_matrix.pckl'
	args.features_path = f'{args.sys_path}/predict_features.pckl'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	predict(args)


if __name__ == '__main__':
	args = initialize_arguments()
	predict(args)
