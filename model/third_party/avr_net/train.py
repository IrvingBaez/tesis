import argparse
import os
import torch
import torch.nn as nn
from shutil import rmtree
from datetime import datetime
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.util import argparse_helper, save_data, get_path, check_system_usage
from .tools.clustering_dataset import ClusteringDataset
from .tools.custom_collator import CustomCollator
from .tools.dataset import CustomDataset
from .tools.feature_extractor import FeatureExtractor
from .tools.losses import MSELoss

def train(args):
	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(f'{args.sys_path}')

	with torch.no_grad():
		train_features = extract_features(args, mode='train', video_proportion=args.video_proportion)
		save_data(train_features, args.train_features_path)
		del train_features

		val_features = extract_features(args, mode='vali', video_proportion=args.video_proportion)
		save_data(val_features, args.val_features_path)
		del val_features

	model, optimizer, scheduler = load_model(args)
	criterion = MSELoss()

	start_epoch, train_losses, train_accs, val_losses, val_accs = load_checkpoint(model, optimizer, scheduler, args)
	total_epochs = start_epoch + args.epochs

	train_loader = load_data(args, mode='train', batch_size=512)
	val_loader = load_data(args, mode='val', batch_size=2048)

	epoch = start_epoch
	for epoch in tqdm(range(start_epoch, total_epochs), initial=start_epoch, total=total_epochs, desc='Epochs', disable=args.disable_pb):
		# ======================== Training Phase ========================
		model.train()
		train_loss = 0.0
		train_accuracy = 0.0

		for batch in tqdm(train_loader, desc='Training phase', leave=False, disable=args.disable_pb):
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

		check_system_usage()
		del output, loss, accuracy, train_accuracy, train_loss
		# ======================== Validation Phase ========================
		model.eval()
		val_loss = 0.0
		val_accuracy = 0.0

		with torch.no_grad():
			for batch in tqdm(val_loader, desc='Validation phase', leave=False, disable=args.disable_pb):
				output = model(batch['video'], batch['audio'], batch['task_full'])
				batch['target'] = batch['target'].to(output.device)

				loss = criterion(output, batch['target'])

				accuracy = ((output > 0.5) == batch['target']).float().mean()
				val_accuracy += accuracy / len(val_loader)
				val_loss += loss / len(val_loader)

		val_losses.append(val_loss.detach().item())
		val_accs.append(val_accuracy.detach().item())

		check_system_usage()
		del output, loss, accuracy, val_accuracy, val_loss
		# ======================== Saving data ========================
		timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
		checkpoint_path = f'{args.checkpoint_dir}/{timestamp}_epoch_{epoch:05d}.ckpt'

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

	description = {
		'Self Attention': 	'TransformerClsToken',
		'Cross Attention': 	'Fusion',
		'video_proportion': args.video_proportion,
		'epochs': 					total_epochs,
		'Optimizer':				'SDG',
		'base LR': 					args.learning_rate,
		'LR Step size': 		args.step_size,
		'LR gamma': 				args.gamma,
		'Momentum':					args.momentum,
		'weight_decay': 		args.weight_decay,
	}
	description = '\n'.join([f'{key}: {value}' for key, value in description.items()])

	timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
	plot_acc_and_loss(train_accs, train_losses, val_accs, val_losses,f'{args.checkpoint_dir}/{timestamp}_epoch_{epoch:05d}.png', description)


def extract_features(args, mode, video_proportion=1):
	feature_extractor = FeatureExtractor()
	feature_extractor = nn.DataParallel(feature_extractor)
	feature_extractor.to(args.device)

	if mode == 'train':
		config = args.train_dataset_config
	else:
		config = args.val_dataset_config

	dataset = CustomDataset(config, training=True, video_proportion=video_proportion, disable_pb=args.disable_pb)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True, drop_last=False, collate_fn=CustomCollator())
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
	model = Attention_AVRNet()
	# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model= nn.DataParallel(model)
	model.to(args.device)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	# optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
	ax1.set_title('Loss')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Value')
	ax1.legend()
	ax1.grid(True)

	ax2.plot(train_acc, label='Train Accuracy')
	ax2.plot(val_acc, label='Validation Accuracy')
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


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	# DATA CONFIGURATION
	parser.add_argument('--video_ids',			type=str,	help='Video ids separated by commas')
	parser.add_argument('--videos_path',		type=str,	help='Path to the videos to work with')
	parser.add_argument('--waves_path',			type=str,	help='Path to the waves, already denoised')
	parser.add_argument('--labs_path',			type=str,	help='Path to the lab files with voice activity detection info')
	parser.add_argument('--frames_path',		type=str,	help='Path to the face frames already cropped and aligned')
	parser.add_argument('--tracks_path',		type=str,	help='Path to the csv files containing the active speaker detection info')
	parser.add_argument('--rttms_path', 		type=str,	help='Path to the rttm files containing detection ground truth')
	parser.add_argument('--sys_path',				type=str,	help='Path to the folder where to save all the system outputs')

	# TRAINING CONFIGURATION
	# parser.add_argument('--gpu_batch_size',	type=int,	help='Training batch size per GPU', default=4)
	parser.add_argument('--learning_rate',		type=float,	help='Training base learning rate', default=0.001)
	parser.add_argument('--momentum',					type=float,	help='Training momentum for SDG optimizer', default=0.05)
	parser.add_argument('--weight_decay',			type=float,	help='Training weight decay for SDG optimizer', default=0.0001)
	parser.add_argument('--step_size',				type=int,	help='Training stepsize for StepLR scheduler', default=5)
	parser.add_argument('--gamma',						type=float,	help='Training gamma for StepLR scheduler', default=0.5)
	parser.add_argument('--video_proportion', type=float, help='Percentage of videos to use in training and validation')
	parser.add_argument('--epochs', 					type=int, help='Epochs to add to the training of the checkpoint', default=100)
	parser.add_argument('--disable_pb', 			action='store_true', help='If true, hides progress bars')

	# MODEL CONFIGURATION
	parser.add_argument('--relation_layer', type=str, help='Type of relation to use', default='original')
	parser.add_argument('--checkpoint', 		type=str,	help='Path of checkpoint to continue training', default=None)

	args = argparse_helper(parser, **kwargs)

	args.train_dataset_config = {
		'video_ids':	 args.video_ids.split(','),
		'waves_path':	 get_path('waves_path', denoiser='dihard18'),
		'rttms_path':	 get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':	 get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path': get_path('asd_path', asd_detector='ground_truth') + '/tracklets'
	}

	with open(f'dataset/split/val.list', 'r') as file:
		val_video_ids = file.read().split('\n')

	args.val_dataset_config = {
		'video_ids':	 val_video_ids,
		'waves_path':	 get_path('waves_path', denoiser='dihard18'),
		'rttms_path':	 get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':	 get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path': get_path('asd_path', asd_detector='ground_truth') + '/tracklets'
	}

	args.data_type = 'train'
	args.video_ids = args.video_ids.split(',')
	args.checkpoint_dir = f'model/third_party/avr_net/checkpoints/{datetime.now().strftime("%Y_%m_%d %H:%M:%S")}'
	args.train_features_path = f'{args.sys_path}/train_features.pckl'
	args.val_features_path = f'{args.sys_path}/val_features.pckl'
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	os.makedirs(args.checkpoint_dir, exist_ok=True)

	# COMENTADO PARA ENTRENAR ATENCIÓN DESDE 0
	# if args.checkpoint is None:
	# 	args.checkpoint = CONFIG['checkpoint']

	if torch.cuda.is_available():
		args.world_size = torch.cuda.device_count()
	else:
		args.world_size = 1

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
