import argparse
import os
import csv
import torch
import torch.nn as nn
from datetime import datetime
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.util import argparse_helper, save_data, get_path
from .tools.clustering_dataset import ClusteringDataset
from .tools.train_dataset import TrainDataset
from .tools.custom_collator import CustomCollator
from .tools.dataset import CustomDataset
from .tools.feature_extractor import FeatureExtractor
from .tools.losses import MSELoss

def train(args):
	os.makedirs(f'{args.sys_path}', exist_ok=True)

	with torch.no_grad():
		print(f'======================== Extracting Features for {args.train_features_path} ========================')
		if not os.path.exists(args.train_features_path):
			train_features = extract_features(args, mode='train', video_proportion=args.video_proportion)
			save_data(train_features, args.train_features_path)
			del train_features

		print(f'======================== Extracting Features for {args.val_features_path} ========================')
		if not os.path.exists(args.val_features_path):
			val_features = extract_features(args, mode='vali', video_proportion=args.val_video_proportion)
			save_data(val_features, args.val_features_path)
			del val_features
		print(f'======================== Features Extracted ========================')

	model, optimizer, scheduler = load_model(args)
	# criterion = MSELoss()
	criterion = nn.BCELoss()

	start_epoch, train_losses, train_accs, val_losses, val_accs = load_checkpoint(model, optimizer, scheduler, args)
	total_epochs = start_epoch + args.epochs

	train_loader = load_data(args, mode='train', batch_size=1024)
	val_loader = load_data(args, mode='val', batch_size=4096)

	epoch = start_epoch

	for param in model.module.relation_layer.parameters():
		param.requires_grad = False

	for epoch in tqdm(range(start_epoch, total_epochs), initial=start_epoch, total=total_epochs, desc='Epochs'):
		# ======================== Training Phase ========================
		model.train()
		train_loss = 0.0
		train_accuracy = 0.0

		if epoch == args.frozen_epochs:
			print(f'======================== Unfreezing relation layer parameters at epoch {epoch} ========================')
			for param in model.module.relation_layer.parameters():
				param.requires_grad = True

		for batch in tqdm(train_loader, desc='Training phase', leave=False, disable=args.disable_pb, mininterval=30.0):
			output = model(batch['video'], batch['audio'], batch['task_full'])
			batch['target'] = batch['target'].to(output.device)

			loss = criterion(output, batch['target'])

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			accuracy = ((output > 0.5) == batch['target']).float().mean()
			train_accuracy += accuracy / len(train_loader)
			train_loss += loss / len(train_loader)

		scheduler.step()
		train_losses.append(train_loss.detach().item())
		train_accs.append(train_accuracy.detach().item())

		del output, loss, accuracy, train_accuracy, train_loss
		# ======================== Validation Phase ========================
		model.eval()
		val_loss = 0.0
		val_accuracy = 0.0

		with torch.no_grad():
			for batch in tqdm(val_loader, desc='Validation phase', leave=False, disable=args.disable_pb, miniters=len(val_loader)//100):
				output = model(batch['video'], batch['audio'], batch['task_full'])
				batch['target'] = batch['target'].to(output.device)

				loss = criterion(output, batch['target'])

				accuracy = ((output > 0.5) == batch['target']).float().mean()
				val_accuracy += accuracy / len(val_loader)
				val_loss += loss / len(val_loader)

		val_losses.append(val_loss.detach().item())
		val_accs.append(val_accuracy.detach().item())

		# check_system_usage()
		del output, loss, accuracy, val_accuracy, val_loss
		# ======================== Saving data ========================
		timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
		checkpoint_path = f'{args.checkpoint_dir}/{timestamp}_epoch_{epoch:05d}.ckpt'

		previous_checkpoint = sorted(glob(f'{args.checkpoint_dir}/*.ckpt'))
		if previous_checkpoint:
			previous_checkpoint = previous_checkpoint[-1]

		save_data({
			'model_state_dict':				model.state_dict(),
			'optimizer_state_dict': 	optimizer.state_dict(),
			'scheduler_state_dict':		scheduler.state_dict(),
			'train_losses': 					train_losses,
			'train_accs': 						train_accs,
			'val_losses': 						val_losses,
			'val_accs': 							val_accs,
			'epoch': 									epoch
		}, checkpoint_path)

		if previous_checkpoint:
			os.remove(previous_checkpoint)

	description = {
		'Self Attention': 				args.self_attention,
		'Self Attention Dropout':	args.self_attention_dropout,
		'Cross Attention': 				args.cross_attention,
		'Video Proportion': 			args.video_proportion,
		'Frames Aligned': 				args.aligned,
		'Max Frames': 						args.max_frames,
		'Epochs': 								total_epochs,
		'Frozen Epochs': 					args.frozen_epochs,
		'Optimizer':							'SGD',
		'Base LR': 								args.learning_rate,
		'LR Step Size': 					args.step_size,
		'LR Gamma': 							args.gamma,
		'Momentum':								args.momentum,
		'Weight Decay': 					args.weight_decay,
	}
	description = '\n'.join([f'{key}: {value}' for key, value in description.items()])

	timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
	plot_acc_and_loss(train_accs, train_losses, val_accs, val_losses,f'{args.checkpoint_dir}/{timestamp}_epoch_{epoch:05d}.png', description)

	return checkpoint_path


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
	print(f'Using {len(features)} data pairs for training')

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
		# dataset = TrainDataset(args.train_features_path, args.train_dataset_config['video_ids'], video_proportion=args.video_proportion, disable_pb=args.disable_pb)
		dataset = ClusteringDataset(args.train_features_path, args.disable_pb)

	if mode=='val':
		dataset = ClusteringDataset(args.val_features_path, args.disable_pb)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())

	return dataloader


def load_model(args):
	model = Attention_AVRNet(args.self_attention, args.cross_attention, dropout=args.self_attention_dropout)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model= nn.DataParallel(model)
	model.to(args.device)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	return model, optimizer, scheduler


def load_checkpoint(model, optimizer, schedueler, args):
	start_epoch, train_losses, train_accs, val_losses, val_accs = 0, [], [], [], []

	if not args.checkpoint:
		print(f'No checkpoint provided, using default initialization')
		return start_epoch, train_losses, train_accs, val_losses, val_accs

	if not os.path.isfile(args.checkpoint):
		print(f'Checkpoint not found at: {args.checkpoint}, using default initialization')
		return start_epoch, train_losses, train_accs, val_losses, val_accs

	checkpoint = torch.load(args.checkpoint)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	schedueler.load_state_dict(checkpoint['scheduler_state_dict'])

	start_epoch 	= checkpoint['epoch'] + 1
	train_losses	= checkpoint['train_losses']
	train_accs 		= checkpoint['train_accs']
	val_losses 		= checkpoint['val_losses']
	val_accs 			= checkpoint['val_accs']

	print(f'Loading checkpoint from {args.checkpoint}, resuming training from epoch {start_epoch}')

	return start_epoch, train_losses, train_accs, val_losses, val_accs


def plot_acc_and_loss(train_acc, train_loss, val_acc, val_loss, save_path, description=''):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.plot(train_loss, label='Train Loss')
	ax1.plot(val_loss, label='Validation Loss')
	ax1.axhline(0.6013, color='darkorange', linestyle='--', linewidth=1.5, label='Val Ref')
	ax1.axhline(1.0520, color='darkblue', linestyle='--', linewidth=1.5, label='Train Ref')
	ax1.set_title('Loss')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Value')
	ax1.legend()
	ax1.grid(True)

	ax2.plot(train_acc, label='Train Accuracy')
	ax2.plot(val_acc, label='Validation Accuracy')
	ax2.axhline(0.8145, color='darkorange', linestyle='--', linewidth=1.5, label='Val Ref')
	ax2.axhline(0.7739, color='darkblue', linestyle='--', linewidth=1.5, label='Train Ref')
	ax2.set_title('Accuracy')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Value')
	ax2.legend()
	ax2.grid(True)

	config_text = description

	plt.subplots_adjust(left=0.25)
	plt.figtext(0.02, 0.5, config_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.5))

	plt.tight_layout(rect=[0.15, 0, 1, 1])
	plt.savefig(save_path)

	csv_save_path = os.path.splitext(save_path)[0] + '.csv'
	with open(csv_save_path, 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		# Write the header
		csvwriter.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
		# Write the data for each epoch
		num_epochs = len(train_acc)
		for epoch in range(num_epochs):
			row = [epoch + 1, train_loss[epoch], val_loss[epoch], train_acc[epoch], val_acc[epoch]]
			csvwriter.writerow(row)



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
