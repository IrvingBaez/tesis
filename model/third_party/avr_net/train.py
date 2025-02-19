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
from glob import glob
from sklearn.metrics import precision_recall_fscore_support as score

from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.util import argparse_helper, save_data, get_path
from .tools.clustering_dataset import ClusteringDataset
from .tools.train_dataset import TrainDataset
from .tools.custom_collator import CustomCollator
from .tools.losses import MSELoss
from .feature_extraction import main as extract_features


def train(args):
	args.train_features_path, args.val_features_path = extract_features(
		sys_path=args.sys_path,
		aligned=args.aligned,
		max_frames=args.max_frames,
		disable_pb=args.disable_pb
	)

	model, optimizer, scheduler = load_model(args)
	# criterion = MSELoss()
	criterion = nn.BCELoss()

	start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args)
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

		train_targets = []
		train_outputs = []

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

			train_targets += batch['target'].detach().cpu().numpy().tolist()
			train_outputs += (output.detach() > 0.5).cpu().numpy().tolist()

		scheduler.step()
		metrics['train_losses'].append(train_loss.detach().item())
		metrics['train_accuracies'].append(train_accuracy.detach().item())

		precision, recall, fscore, support = score(train_targets, train_outputs, average='binary')
		metrics['train_precisions'].append(precision)
		metrics['train_recalls'].append(recall)
		metrics['train_f1s'].append(fscore)

		del output, loss, accuracy, train_accuracy, train_loss, train_targets, train_outputs
		# ======================== Validation Phase ========================
		model.eval()
		val_loss = 0.0
		val_accuracy = 0.0

		val_targets = []
		val_outputs = []

		with torch.no_grad():
			for batch in tqdm(val_loader, desc='Validation phase', leave=False, disable=args.disable_pb, miniters=len(val_loader)//100):
				output = model(batch['video'], batch['audio'], batch['task_full'])
				batch['target'] = batch['target'].to(output.device)

				loss = criterion(output, batch['target'])

				accuracy = ((output > 0.5) == batch['target']).float().mean()
				val_accuracy += accuracy / len(val_loader)
				val_loss += loss / len(val_loader)

				val_targets += batch['target'].detach().cpu().numpy().tolist()
				val_outputs += (output.detach() > 0.5).cpu().numpy().tolist()

		metrics['val_losses'].append(val_loss.detach().item())
		metrics['val_accuracies'].append(val_accuracy.detach().item())

		precision, recall, fscore, support = score(val_targets, val_outputs, average='binary')
		metrics['val_precisions'].append(precision)
		metrics['val_recalls'].append(recall)
		metrics['val_f1s'].append(fscore)

		# check_system_usage()
		del output, loss, accuracy, val_accuracy, val_loss, val_targets, val_outputs
		# ======================== Saving data ========================
		timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
		checkpoint_path = f'{args.checkpoint_dir}/{timestamp}_epoch_{epoch:05d}.ckpt'

		previous_checkpoint = sorted(glob(f'{args.checkpoint_dir}/*.ckpt'))
		if previous_checkpoint:
			previous_checkpoint = previous_checkpoint[-1]

		checkpoint = {
			'model_state_dict':				model.state_dict(),
			'optimizer_state_dict': 	optimizer.state_dict(),
			'scheduler_state_dict':		scheduler.state_dict(),
			'metrics': 								metrics,
			'epoch': 									epoch
		}

		save_data(checkpoint, checkpoint_path)

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

		plot_acc_and_loss(metrics, args.checkpoint_dir, epoch, description)

		if previous_checkpoint:
			os.remove(previous_checkpoint)

	return checkpoint_path


def load_data(args, mode, batch_size=256):
	if mode=='train':
		dataset = ClusteringDataset(args.train_features_path, args.disable_pb, video_proportion=args.video_proportion)

	if mode=='val':
		dataset = ClusteringDataset(args.val_features_path, args.disable_pb, video_proportion=args.val_video_proportion)

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())

	return dataloader


def load_model(args):
	model = Attention_AVRNet(args.self_attention, args.cross_attention, dropout=args.self_attention_dropout)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model= nn.DataParallel(model)
	model.to(args.device)

	# optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	optimizer = Adam(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	return model, optimizer, scheduler


def load_checkpoint(model, optimizer, schedueler, args):
	start_epoch = 0
	metrics = {
		'train_losses': 		[],
		'train_accuracies': [],
		'train_precisions': [],
		'train_recalls': 		[],
		'train_f1s': 				[],
		'val_losses': 			[],
		'val_accuracies': 	[],
		'val_precisions': 	[],
		'val_recalls': 			[],
		'val_f1s': 					[]
	}

	if not args.checkpoint:
		print(f'No checkpoint provided, using default initialization')
		return start_epoch, metrics

	if not os.path.isfile(args.checkpoint):
		print(f'Checkpoint not found at: {args.checkpoint}, using default initialization')
		return start_epoch, metrics

	checkpoint = torch.load(args.checkpoint)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	schedueler.load_state_dict(checkpoint['scheduler_state_dict'])

	start_epoch = checkpoint['epoch'] + 1
	metrics			= checkpoint['metrics']

	print(f'Loading checkpoint from {args.checkpoint}, resuming training from epoch {start_epoch}')

	return start_epoch, metrics


def plot_acc_and_loss(metrics, save_dir, epoch, description=''):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.plot(metrics['train_losses'], label='Train Loss')
	ax1.plot(metrics['val_losses'], label='Validation Loss')
	ax1.axhline(0.6013, color='darkorange', linestyle='--', linewidth=1.5, label='Val Ref')
	ax1.axhline(1.0520, color='darkblue', linestyle='--', linewidth=1.5, label='Train Ref')
	ax1.set_title('Loss')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Value')
	ax1.legend()
	ax1.grid(True)

	ax2.plot(metrics['train_accuracies'], label='Train Accuracy')
	ax2.plot(metrics['val_accuracies'], label='Validation Accuracy')
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
	plt.savefig(f'{save_dir}/metrics.png')
	plt.close(fig)

	fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	ax1.plot(metrics['train_f1s'], 				label='Train F1')
	ax1.plot(metrics['train_precisions'], label='Train Precision')
	ax1.plot(metrics['train_recalls'], 		label='Train Recall')
	ax1.set_title('Training')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Value')
	ax1.legend()
	ax1.grid(True)

	ax2.plot(metrics['val_f1s'], 					label='Val F1')
	ax2.plot(metrics['val_precisions'], 	label='Val Precision')
	ax2.plot(metrics['val_recalls'], 			label='Val Recall')
	ax2.set_title('Validation')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Value')
	ax2.legend()
	ax2.grid(True)

	plt.savefig(f'{save_dir}/f1s.png')
	plt.close(fig2)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Attention AVR-Net training")

	# DATA CONFIGURATION
	parser.add_argument('--aligned', 								action='store_true', help='Wether or not to use alined frames')
	parser.add_argument('--max_frames', 						type=int, help='How many frames to use in self-attention')
	parser.add_argument('--video_proportion', 			type=float, help='Percentage of available videos to use in training')
	parser.add_argument('--val_video_proportion', 	type=float, help='Percentage of available videos to use in validation')

	# TRAINING CONFIGURATION
	# parser.add_argument('--gpu_batch_size',	type=int,	help='Training batch size per GPU', default=4)
	parser.add_argument('--learning_rate',					type=float,	help='Training base learning rate', 															default=0.0005)
	parser.add_argument('--momentum',								type=float,	help='Training momentum for SDG optimizer', 											default=0.05)
	parser.add_argument('--weight_decay',						type=float,	help='Training weight decay for SDG optimizer', 									default=0.0001)
	parser.add_argument('--step_size',							type=int,		help='Training stepsize for StepLR scheduler', 										default=5)
	parser.add_argument('--gamma',									type=float,	help='Training gamma for StepLR scheduler', 											default=0.5)
	parser.add_argument('--epochs', 								type=int, 	help='Epochs to add to the training of the checkpoint', 					default=10)
	parser.add_argument('--frozen_epochs', 					type=int, 	help='Epochs to train without updating relation network weights', default=0)
	parser.add_argument('--self_attention', 				type=str, 	help='Self attention method to marge available frame features', 	default='')
	parser.add_argument('--self_attention_dropout', type=float, help='Dropout used in self-attention transformer', 								default=0.1)
	parser.add_argument('--cross_attention', 				type=str, 	help='Cross attention method to marge frame and audio features', 	default='')
	parser.add_argument('--disable_pb', 						action='store_true', help='If true, hides progress bars')

	# MODEL CONFIGURATION
	parser.add_argument('--relation_layer', type=str, help='Type of relation to use', default='original')
	parser.add_argument('--checkpoint', 		type=str,	help='Path of checkpoint to continue training', default=None)

	args = argparse_helper(parser, **kwargs)

	args.sys_path					= 'model/third_party/avr_net/features'
	args.checkpoint_dir		= f'model/third_party/avr_net/checkpoints/{datetime.now().strftime("%Y_%m_%d %H:%M:%S")}'
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	os.makedirs(args.checkpoint_dir, exist_ok=True)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	return train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
