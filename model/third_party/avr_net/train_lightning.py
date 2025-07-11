import argparse
import copy
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SequentialSampler
from pathlib import Path

from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.third_party.avr_net.tools.write_rttms import main as write_rttms
from model.avd.score_avd import main as score_avd
from model.util import argparse_helper, get_path, save_data
from .feature_extraction import main as extract_features
from .tools.clustering_dataset import ClusteringDataset
from .tools.custom_collator import CustomCollator
from .tools.contrastive_loss import ContrastiveLoss

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall


class Lightning_Attention_AVRNet(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.save_hyperparameters()

		if args.loss_fn == 'bce':
			self.loss_fn = F.binary_cross_entropy
		elif args.loss_fn == 'mse':
			self.loss_fn = F.mse_loss
		elif args.loss_fn == 'contrastive':
			self.loss_fn = ContrastiveLoss(pos_margin=args.pos_margin, neg_margin=args.neg_margin)
		else:
			raise ValueError(f"loss_fn must be 'bce', 'mse' or 'contrastive' not '{args.loss_fn}'")

		self.model = Attention_AVRNet(self.args.self_attention, self.args.cross_attention, dropout=self.args.self_attention_dropout)
		self.model.freeze_relation()

		if 'fine_tunning' not in self.args:
			self.args.fine_tunning = False

		if self.args.fine_tunning:
			self.model.fine_tunning()

		self.cos = nn.CosineSimilarity()
		self.train_cos_stats = {'video': [], 'audio': [], 'target': []}
		self.val_cos_stats = {'video': [], 'audio': [], 'target': []}

		self.metric = BinaryF1Score()
		self.metric_recall = BinaryRecall()
		self.metric_precision = BinaryPrecision()

		self.train_predictions = None
		self.validation_predictions = None


	def configure_optimizers(self):
		print(f"CONFIGURING OPTIMIZER WITH LR: {self.args.learning_rate}")

		if self.args.optimizer == 'sgd':
			optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
		elif self.args.optimizer == 'adam':
			optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
		else:
			raise ValueError(f"optimizer must be 'sgd' or 'adam' not '{self.args.optimizer}'")

		lr_scheduler = StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

		return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


	def on_train_epoch_start(self):
		self.train_predictions = []
		self.train_cos_stats = {'video': [], 'audio': [], 'target': []}

		if not self.args.fine_tunning and self.current_epoch >= self.args.frozen_epochs:
			self.model.unfreeze_relation()


	def training_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		scores = self.model(video, audio, task_full)

		loss = self.loss_fn(scores, target)

		accuracy = ((scores > 0.5) == target).float().mean()
		fscore = self.metric(scores, target)

		self.train_cos_stats = self.cos_batch_stats(video, audio, target, self.train_cos_stats)
		# self.cosine_stats(video, audio, target, mode='train')

		self.log("loss/train", 			loss, 			sync_dist=True) #, on_step=True)
		self.log("acc/train", 			accuracy, 	sync_dist=True) #, on_step=True)
		self.log("f1_score/train", 	fscore, 		sync_dist=True) #, on_step=True)

		for video_id, index_a, index_b, score in zip(batch['video_id'], batch['index_a'], batch['index_b'], scores):
			self.train_predictions.append((video_id, index_a, index_b, score.detach().cpu()))

		return loss


	def on_train_epoch_end(self):
		return

		self.write_cos_stats(self.train_cos_stats, 'train')

		similarities = {}

		for video_id in self.args.train_utterance_counts.keys():
			similarities[video_id] = torch.diag_embed(torch.ones([self.args.train_utterance_counts[video_id]]))

		for prediction in self.train_predictions:
			video_id, index_a, index_b, score = prediction

			similarities[video_id][index_a, index_b] = score
			similarities[video_id][index_b, index_a] = score

		self.train_predictions = None

		similarities_path = f'{self.logger.log_dir}/similarities.pth'
		sys_path = f'{self.logger.log_dir}/rttms'

		similarity_data = {'similarities': similarities, 'starts': self.args.train_starts, 'ends': self.args.train_ends}
		save_data(similarity_data, similarities_path, verbose=True, override=True)
		write_rttms(similarities_path=similarities_path, sys_path=sys_path, data_type='train')

		score_avd(data_type='train', sys_path=f'{sys_path}/train.out', output_path=f'{sys_path}/train_scores.out')

		with open(f'{sys_path}/train_scores.out', 'r') as file:
			scores = file.read()

		last_line = scores.split('\n')[-1]
		der = float(last_line.split()[3])

		self.log("der/train", der, 	sync_dist=True)
		shutil.rmtree(sys_path)
		os.remove(similarities_path)


	def on_validation_epoch_start(self):
		self.validation_predictions = []
		self.val_cos_stats = {'video': [], 'audio': [], 'target': []}


	def validation_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		scores = self.model(video, audio, task_full)

		loss = self.loss_fn(scores, target)

		accuracy = 	((scores > 0.5) == target).float().mean()
		fscore = 		self.metric(scores, target)
		recall = 		self.metric_recall(scores, target)
		precision = self.metric_precision(scores, target)

		self.val_cos_stats = self.cos_batch_stats(video, audio, target, self.val_cos_stats)
		# self.cosine_stats(video, audio, target, mode='val')

		self.log("loss/val", 			loss, 			sync_dist=True)
		self.log("acc/val", 			accuracy, 	sync_dist=True)
		self.log("f1_score/val", 	fscore, 		sync_dist=True)
		self.log("recall/val", 		recall, 		sync_dist=True)
		self.log("precision/val", precision, 	sync_dist=True)

		for video_id, index_a, index_b, score in zip(batch['video_id'], batch['index_a'], batch['index_b'], scores):
			self.validation_predictions.append((video_id, index_a, index_b, score.detach().cpu()))

		return loss


	# Calculate DER as part of validation end
	def on_validation_epoch_end(self):
		# self.write_cos_stats(self.val_cos_stats, 'val')

		similarities = {}

		for video_id in self.args.val_utterance_counts.keys():
			similarities[video_id] = torch.diag_embed(torch.ones([self.args.val_utterance_counts[video_id]]))

		for prediction in self.validation_predictions:
			video_id, index_a, index_b, score = prediction

			similarities[video_id][index_a, index_b] = score
			similarities[video_id][index_b, index_a] = score

		self.validation_predictions = None

		similarities_path = f'{self.logger.log_dir}/similarities.pth'
		sys_path = f'{self.logger.log_dir}/rttms'

		similarity_data = {'similarities': similarities, 'starts': self.args.val_starts, 'ends': self.args.val_ends}
		save_data(similarity_data, similarities_path, verbose=True, override=True)
		write_rttms(similarities_path=similarities_path, sys_path=sys_path, data_type='val', ahc_threshold=self.args.ahc_threshold)

		score_avd(data_type='val', sys_path=f'{sys_path}/val.out', output_path=f'{sys_path}/val_scores.out')

		with open(f'{sys_path}/val_scores.out', 'r') as file:
			scores = file.read()

		last_line = scores.split('\n')[-1]
		der = float(last_line.split()[3])

		self.log("der/val", der, 	sync_dist=True)
		shutil.rmtree(sys_path)
		os.remove(similarities_path)


	def on_test_epoch_start(self):
		self.validation_predictions = []
		# self.val_cos_stats = {'video': [], 'audio': [], 'target': []}


	def test_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		scores = self.model(video, audio, task_full)

		loss = self.loss_fn(scores, target)

		accuracy = 	((scores > 0.5) == target).float().mean()
		fscore = 		self.metric(scores, target)
		recall = 		self.metric_recall(scores, target)
		precision = self.metric_precision(scores, target)

		# self.val_cos_stats = self.cos_batch_stats(video, audio, target, self.val_cos_stats)
		# self.cosine_stats(video, audio, target, mode='val')

		self.log("loss/test", 			loss, 			sync_dist=True)
		self.log("acc/test", 				accuracy, 	sync_dist=True)
		self.log("f1_score/test", 	fscore, 		sync_dist=True)
		self.log("recall/test", 		recall, 		sync_dist=True)
		self.log("precision/test", 	precision, 	sync_dist=True)

		for video_id, index_a, index_b, score in zip(batch['video_id'], batch['index_a'], batch['index_b'], scores):
			self.validation_predictions.append((video_id, index_a, index_b, score.detach().cpu()))

		return loss


	# Calculate DER as part of validation end
	def on_test_epoch_end(self):
		# self.write_cos_stats(self.val_cos_stats, 'val')

		similarities = {}

		for video_id in self.args.val_utterance_counts.keys():
			similarities[video_id] = torch.diag_embed(torch.ones([self.args.val_utterance_counts[video_id]]))

		for prediction in self.validation_predictions:
			video_id, index_a, index_b, score = prediction

			similarities[video_id][index_a, index_b] = score
			similarities[video_id][index_b, index_a] = score

		self.validation_predictions = None

		similarities_path = f'{self.logger.log_dir}/similarities_test.pth'
		sys_path = f'{self.logger.log_dir}/rttms'

		similarity_data = {'similarities': similarities, 'starts': self.args.val_starts, 'ends': self.args.val_ends}
		save_data(similarity_data, similarities_path, verbose=True, override=True)
		write_rttms(similarities_path=similarities_path, sys_path=sys_path, data_type='test', ahc_threshold=self.args.ahc_threshold)

		score_avd(data_type='test', sys_path=f'{sys_path}/test.out', output_path=f'{sys_path}/test_scores.out')

		with open(f'{sys_path}/test_scores.out', 'r') as file:
			scores = file.read()

		last_line = scores.split('\n')[-1]
		der = float(last_line.split()[3])

		self.log("der/test", der, 	sync_dist=True)
		shutil.rmtree(sys_path)
		os.remove(similarities_path)


	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		predictions = []
		scores = self.model(batch['video'], batch['audio'], batch['task_full'])

		for video_id, index_a, index_b, score in zip(batch['video_id'], batch['index_a'], batch['index_b'], scores):
			predictions.append((video_id, index_a, index_b, score.cpu()))

		return predictions


	def cos_batch_stats(self, video, audio, target, data_dict):
		B = video.shape[0]
		target = target.reshape(B)

		video_a, video_b = video[:,0,...].reshape(B, 512*49), video[:,1,...].reshape(B, 512*49)
		video_cos = self.cos(video_a, video_b)
		data_dict['video'].extend(video_cos.tolist())

		audio_a, audio_b = audio[:,0,...].reshape(B, 256*49), audio[:,1,...].reshape(B, 256*49)
		audio_cos = self.cos(audio_a, audio_b)
		data_dict['audio'].extend(audio_cos.tolist())

		data_dict['target'].extend(target.tolist())
		return data_dict


	def write_cos_stats(self, data_dict, mode):
		video = np.array(data_dict['video'])
		audio = np.array(data_dict['audio'])
		target = np.array(data_dict['target'])

		pos_video = video[target.nonzero()]
		neg_video = video[(1-target).nonzero()]
		pos_audio = audio[target.nonzero()]
		neg_audio = audio[(1-target).nonzero()]

		self.log(f"pos_vid_cos_mean/{mode}", 	pos_video.mean(), sync_dist=True)
		self.log(f"pos_vid_cos_std/{mode}", 	pos_video.std(), 	sync_dist=True)
		self.log(f"neg_vid_cos_mean/{mode}",	neg_video.mean(),	sync_dist=True)
		self.log(f"neg_vid_cos_std/{mode}", 	neg_video.std(), 	sync_dist=True)
		self.log(f"pos_aud_cos_mean/{mode}", 	pos_audio.mean(), sync_dist=True)
		self.log(f"pos_aud_cos_std/{mode}", 	pos_audio.std(), 	sync_dist=True)
		self.log(f"neg_aud_cos_mean/{mode}", 	neg_audio.mean(), sync_dist=True)
		self.log(f"neg_aud_cos_std/{mode}", 	neg_audio.std(), 	sync_dist=True)



def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Attention AVR-Net training")

	# TRAINING CONFIGURATION
	parser.add_argument('--learning_rate',					type=float,	default=0.001, 		help='Training base learning rate')
	parser.add_argument('--momentum',								type=float, default=0.0,			help='Training momentum for SDG optimizer. Ignored if Adam optimizer is selected.')
	parser.add_argument('--weight_decay',						type=float, default=0.0001,		help='Training weight decay for SDG optimizer')
	parser.add_argument('--step_size',							type=int, 	default=5,				help='Training stepsize for StepLR scheduler')
	parser.add_argument('--gamma',									type=float, default=0.5,			help='Training gamma for StepLR scheduler')
	parser.add_argument('--video_proportion', 			type=float, default=1.0,			help='Percentage of available videos to use in training')
	parser.add_argument('--val_video_proportion', 	type=float, default=1.0,			help='Percentage of available videos to use in validation')
	parser.add_argument('--epochs', 								type=int, 	default=10, 			help='Epochs to add to the training of the checkpoint')
	parser.add_argument('--max_epochs',							type=int, 	default=None, 		help='Force stop after this many epochs', required=False)
	parser.add_argument('--frozen_epochs', 					type=int, 	default=0, 				help='Epochs to train without updating relation network weights')
	parser.add_argument('--self_attention', 				type=str, 	default='',				help='Self attention method to marge available frame features')
	parser.add_argument('--self_attention_dropout', type=float, default=0.1, 			help='Dropout used in self-attention transformer')
	parser.add_argument('--cross_attention', 				type=str, 	default='', 			help='Cross attention method to marge frame and audio features')
	parser.add_argument('--loss_fn', 								type=str, 	default='', 			help='Loss function to use during training')
	parser.add_argument('--pos_margin', 						type=float, default=0.0, 			help='Positive margin for contrastive loss')
	parser.add_argument('--neg_margin', 						type=float, default=1.0, 			help='Negative margin for contrastive loss')
	parser.add_argument('--ahc_threshold', 					type=float, default=0.3, 			help='Threshold distance for the AHC algorithm')
	parser.add_argument('--optimizer', 							type=str, 	default='', 			help='Optimizer to use during training')
	parser.add_argument('--task', 									type=str, 	default='train', 	help='Execution mode, either train or val')
	parser.add_argument('--disable_pb', 						action='store_true', 					help='If true, hides progress bars')
	# DATA CONFIGURATION
	parser.add_argument('--max_frames', 						type=int, 	default=1,				help='How many frames to use in self-attention')
	parser.add_argument('--db_video_mode', 					type=str, 	default='pick_first',	help='Selection mode for video frames in the dataset')
	parser.add_argument('--aligned', 								action='store_true', 					help='Wether or not to use aligned frames')
	parser.add_argument('--balanced',								action='store_true', 					help='Balance positives and negatives examples in training data.')
	# MODEL CONFIGURATION
	parser.add_argument('--checkpoint', 						type=str,		default=None, 		help='Path of checkpoint to continue training.')
	parser.add_argument('--fine_tunning', 					action='store_true', 					help='Freezes all but the last relation layer.')

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

	# Unused?
	args.val_dataset_config = {
		'video_ids':		val_video_ids,
		'waves_path':		get_path('waves_path', denoiser='dihard18'),
		'rttms_path':		get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':		get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path':	get_path('asd_path', asd_detector='ground_truth') + ('/aligned_tracklets' if args.aligned else '/tracklets'),
		'max_frames':		args.max_frames
	}

	assert Path(args.train_dataset_config['waves_path']).exists(), 	f'Data path {args.train_dataset_config['waves_path']} does not exist.'
	assert Path(args.train_dataset_config['rttms_path']).exists(), 	f'Data path {args.train_dataset_config['rttms_path']} does not exist.'
	assert Path(args.train_dataset_config['labs_path']).exists(), 	f'Data path {args.train_dataset_config['labs_path']} does not exist.'
	assert Path(args.train_dataset_config['frames_path']).exists(), f'Data path {args.train_dataset_config['frames_path']} does not exist.'
	assert Path(args.val_dataset_config['waves_path']).exists(), 		f'Data path {args.val_dataset_config['waves_path']} does not exist.'
	assert Path(args.val_dataset_config['rttms_path']).exists(), 		f'Data path {args.val_dataset_config['rttms_path']} does not exist.'
	assert Path(args.val_dataset_config['labs_path']).exists(), 		f'Data path {args.val_dataset_config['labs_path']} does not exist.'
	assert Path(args.val_dataset_config['frames_path']).exists(), 	f'Data path {args.val_dataset_config['frames_path']} does not exist.'

	args.sys_path 			= 'model/third_party/avr_net/features'
	args.checkpoint_dir = f'model/third_party/avr_net/checkpoints/'

	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	return args


def train(args):
	torch.set_float32_matmul_precision('high')

	train_loader, eval_loader = create_dataset(args)

	args.val_utterance_counts = eval_loader.dataset.utterance_counts
	args.val_starts = eval_loader.dataset.starts()
	args.val_ends = eval_loader.dataset.ends()

	args.train_utterance_counts = train_loader.dataset.utterance_counts
	args.train_starts = train_loader.dataset.starts()
	args.train_ends = train_loader.dataset.ends()

	# TODO: Move to load_model
	if args.checkpoint:
		print(f'Loading checkpoint from: {args.checkpoint}')
		model = Lightning_Attention_AVRNet(clean_up_args(args))
		state_dict = torch.load(args.checkpoint)['state_dict']
		model.load_state_dict(state_dict)
	else:
		print('Loading default model weights')
		model = Lightning_Attention_AVRNet(clean_up_args(args))

	trainer = pl.Trainer(
		accelerator="gpu", devices=1, strategy="auto", # auto ddp_find_unused_parameters_true ddp
		default_root_dir=args.checkpoint_dir,
		min_epochs=args.epochs,
		max_epochs=args.max_epochs,
		callbacks=[
			ModelCheckpoint(monitor="der/val", mode="min"),
			EarlyStopping(monitor="der/val", mode="min")
		],
	)

	if args.task == 'train':
		trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
	elif args.task == 'val':
		assert args.checkpoint is not None, 'Cannot eval without a given checkpoint'
		trainer.validate(model=model, dataloaders=eval_loader)
	elif args.task == 'test':
		assert args.checkpoint is not None, 'Cannot test without a given checkpoint'
		trainer.test(model=model, dataloaders=eval_loader)


def clean_up_args(args):
		allowed_args = ['loss_fn', 'add_contrastive', 'val_starts', 'val_ends', 'val_utterance_counts', 'train_starts', 'train_ends', 'train_utterance_counts', 'self_attention', 'cross_attention', 'self_attention_dropout', 'fine_tunning', 'learning_rate', 'weight_decay', 'momentum', 'optimizer', 'step_size', 'gamma', 'frozen_epochs', 'pos_margin', 'neg_margin', 'ahc_threshold']
		current_args = [key for key in args.__dict__.keys()]

		args_copy = copy.deepcopy(args)

		for key in current_args:
			if key not in allowed_args:
				args_copy.__dict__.pop(key)

		not_empty = ['val_starts', 'val_ends', 'val_utterance_counts', 'train_starts', 'train_ends', 'train_utterance_counts']

		for element in not_empty:
			if element not in args_copy.__dict__.keys():
				args_copy.__dict__[element] = None

		return args_copy


def create_dataset(args):
	args.train_features_path, args.val_features_path, args.test_features_path = extract_features(
		sys_path=args.sys_path,
		aligned=args.aligned,
		max_frames=args.max_frames,
		db_video_mode=args.db_video_mode,
		disable_pb=args.disable_pb
	)

	train_loader = load_data(args, mode='train', workers=11, batch_size=64)
	eval_loader = load_data(args, mode=args.task, workers=2, batch_size=512)

	return train_loader, eval_loader


def load_data(args, mode, workers, batch_size=256):
	if mode=='train':
		dataset = ClusteringDataset(args.train_features_path, args.disable_pb, video_proportion=args.video_proportion, balanced=args.balanced)
	elif mode=='val':
		dataset = ClusteringDataset(args.val_features_path, args.disable_pb, video_proportion=args.val_video_proportion)
	elif mode=='test':
		print('Ignoring val_video_proportion, testing with proportion 1.0')
		dataset = ClusteringDataset(args.test_features_path, args.disable_pb, video_proportion=1.0)
	else:
		raise ValueError(f"Dataset mode '{mode}' not recognized")

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

	return dataloader


def load_model(args):
	model = Attention_AVRNet(args.self_attention, args.cross_attention)
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model= nn.DataParallel(model)
	model.to(args.device)
	model.train()

	optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

	return model, optimizer, scheduler


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	return train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
