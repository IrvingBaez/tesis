import argparse
import os
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.util import argparse_helper, save_data, get_path
from .tools.clustering_dataset import ClusteringDataset
from .tools.train_dataset import TrainDataset
from .tools.custom_collator import CustomCollator
from .tools.dataset import CustomDataset
from .tools.feature_extractor import FeatureExtractor

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary

class Lightning_Attention_AVRNet(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.save_hyperparameters()

		self.model = Attention_AVRNet(self.args.self_attention, self.args.cross_attention, dropout=self.args.self_attention_dropout)
		self.criterion = nn.BCELoss()


	def configure_optimizers(self):
		optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
		return optimizer


	def forward(self, video, audio, task):
		self.model(video, audio, task)


	# TODO: Freeze parameters
	def training_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		output = self.model(video, audio, task_full)

		loss = self.criterion(output, target)
		accuracy = ((output > 0.5) == target).float().mean()
		self.log_dict({"train_loss": loss, "train_acc": accuracy})

		return loss


	def validation_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		output = self.model(video, audio, task_full)

		loss = self.criterion(output, target)
		accuracy = ((output > 0.5) == target).float().mean()
		self.log_dict({"val_loss": loss, "val_acc": accuracy})


def train(args):
	model = Lightning_Attention_AVRNet(args)
	train_loader, val_loader = create_dataset(args)

	trainer = pl.Trainer(
		accelerator="gpu", devices=4, strategy="ddp",
		default_root_dir=args.checkpoint_dir,
		callbacks=[
			EarlyStopping(monitor="val_loss", mode="min"),
			ModelSummary(max_depth=2)
		],
		profiler="simple"
	)

	trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def create_dataset(args):
	os.makedirs(f'{args.sys_path}', exist_ok=True)

	with torch.no_grad():
		print('======================== Extracting Features ========================')
		if not os.path.exists(args.train_features_path):
			train_features = extract_features(args, mode='train', video_proportion=args.video_proportion)
			save_data(train_features, args.train_features_path)
			del train_features

		if not os.path.exists(args.val_features_path):
			val_features = extract_features(args, mode='vali', video_proportion=0.1)
			save_data(val_features, args.val_features_path)
			del val_features
		print('======================== Features Extracted ========================')

	train_loader = load_data(args, mode='train', batch_size=2048)
	val_loader = load_data(args, mode='val', batch_size=4096)

	return train_loader, val_loader


def extract_features(args, mode, video_proportion=1.0):
	feature_extractor = FeatureExtractor()
	feature_extractor = nn.DataParallel(feature_extractor)
	feature_extractor.to(args.device)

	if mode == 'train':
		config = args.train_dataset_config
	else:
		config = args.val_dataset_config

	dataset = CustomDataset(config, training=True, video_proportion=video_proportion, disable_pb=args.disable_pb)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=CustomCollator())
	dataloader = tqdm(dataloader, desc='Extracting features', disable=args.disable_pb)
	print(len(dataset))

	feature_list = []
	with torch.no_grad():
		for batch in dataloader:
			features = {}

			feat_audio, feat_video = feature_extractor(batch['audio'], batch['frames'])

			features['feat_audio'] 	= feat_audio
			features['feat_video'] 	= feat_video
			features['video'] 			=	list(batch['meta']['video']),
			features['start'] 			=	batch['meta']['start'],
			features['end'] 				=	batch['meta']['end'],
			features['trackid'] 		=	batch['meta']['trackid'],
			features['visible'] 		=	batch['visible'],
			features['losses'] 			=	{}

			if 'targets' in batch.keys():
				features['targets'] = batch['targets']

			for key, value in features.items():
				if isinstance(value, torch.Tensor):
					features[key] = value.cpu()

			feature_list.append(features)

	features = merge_features(feature_list)

	return features


def merge_features(dicts):
	features = dicts[0]

	for key, value in features.items():
		if isinstance(value, tuple):
			features[key] = value[0]

	for batch in dicts[1:]:
		for key, value in batch.items():
			if isinstance(value, tuple):
				value = value[0]

			if isinstance(value, list):
				features[key].extend(value)
			elif isinstance(value, torch.Tensor):
				features[key] = torch.cat((features[key], value))
			elif isinstance(value, dict):
				features[key] = merge_features([features[key], value])

	for key, value in features.items():
		if isinstance(value, torch.Tensor):
			features[key] = value.cpu()

	return features


def load_data(args, mode, batch_size=256):
	if mode=='train':
		dataset = ClusteringDataset(args.train_features_path, args.disable_pb)

	if mode=='val':
		dataset = ClusteringDataset(args.val_features_path, args.disable_pb)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())

	return dataloader


def load_model(args):
	model = Attention_AVRNet(args.self_attention, args.cross_attention)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model= nn.DataParallel(model)
	model.to(args.device)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	return model, optimizer, scheduler


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Attention AVR-Net training")

	# DATA CONFIGURATION
	parser.add_argument('--aligned', 					action='store_true', help='Wether or not to use alined frames')
	parser.add_argument('--max_frames', 			type=int, help='How many frames to use in self-attention')

	# TRAINING CONFIGURATION
	# parser.add_argument('--gpu_batch_size',	type=int,	help='Training batch size per GPU', default=4)
	parser.add_argument('--learning_rate',					type=float,	help='Training base learning rate', default=0.0005)
	parser.add_argument('--momentum',								type=float,	help='Training momentum for SDG optimizer', default=0.05)
	parser.add_argument('--weight_decay',						type=float,	help='Training weight decay for SDG optimizer', default=0.0001)
	parser.add_argument('--step_size',							type=int,		help='Training stepsize for StepLR scheduler', default=5)
	parser.add_argument('--gamma',									type=float,	help='Training gamma for StepLR scheduler', default=0.5)
	parser.add_argument('--video_proportion', 			type=float, help='Percentage of available videos to use in training')
	parser.add_argument('--val_video_proportion', 	type=float, help='Percentage of available videos to use in validation')
	parser.add_argument('--epochs', 								type=int, 	help='Epochs to add to the training of the checkpoint', default=10)
	parser.add_argument('--frozen_epochs', 					type=int, 	help='Epochs to train without updating relation network weights', default=0)
	parser.add_argument('--self_attention', 				type=str, 	help='Self attention method to marge available frame features', default='')
	parser.add_argument('--self_attention_dropout', type=float, help='Dropout used in self-attention transformer', default=0.1)
	parser.add_argument('--cross_attention', 				type=str, 	help='Cross attention method to marge frame and audio features', default='')
	parser.add_argument('--disable_pb', 						action='store_true', help='If true, hides progress bars')

	# MODEL CONFIGURATION
	parser.add_argument('--relation_layer', type=str, help='Type of relation to use', default='original')
	parser.add_argument('--checkpoint', 		type=str,	help='Path of checkpoint to continue training', default=None)

	args = argparse_helper(parser, **kwargs)

	with open(f'dataset/split/train.list', 'r') as file:
		args.video_ids = file.read().split('\n')

	args.train_dataset_config = {
		'video_ids':		args.video_ids,
		'waves_path':		get_path('waves_path', denoiser='dihard18'),
		'rttms_path':		get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':		get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path':	get_path('asd_path', asd_detector='ground_truth') + ('/aligned_tracklets' if args.aligned else '/tracklets'),
		'max_frames':		args.max_frames
	}

	with open(f'dataset/split/val.list', 'r') as file:
		val_video_ids = file.read().split('\n')

	args.val_dataset_config = {
		'video_ids':		val_video_ids,
		'waves_path':		get_path('waves_path', denoiser='dihard18'),
		'rttms_path':		get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':		get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path':	get_path('asd_path', asd_detector='ground_truth') + ('/aligned_tracklets' if args.aligned else '/tracklets'),
		'max_frames':		args.max_frames
	}

	assert Path(args.train_dataset_config['waves_path']).exists()
	assert Path(args.train_dataset_config['rttms_path']).exists()
	assert Path(args.train_dataset_config['labs_path']).exists()
	assert Path(args.train_dataset_config['frames_path']).exists()
	assert Path(args.val_dataset_config['waves_path']).exists()
	assert Path(args.val_dataset_config['rttms_path']).exists()
	assert Path(args.val_dataset_config['labs_path']).exists()
	assert Path(args.val_dataset_config['frames_path']).exists()

	args.sys_path 						= 'model/third_party/avr_net/features'
	args.checkpoint_dir 			= f'model/third_party/avr_net/checkpoints/{datetime.now().strftime("%Y_%m_%d %H:%M:%S")}'
	args.train_features_path 	= f'{args.sys_path}/train_features_{int(args.video_proportion * 100)}p_{args.max_frames}f{'_aligned' if args.aligned else ''}.pckl'
	args.val_features_path 		= f'{args.sys_path}/val_features_{int(args.val_video_proportion * 100)}p_{args.max_frames}f{'_aligned' if args.aligned else ''}.pckl'
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	os.makedirs(args.checkpoint_dir, exist_ok=True)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	return train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
