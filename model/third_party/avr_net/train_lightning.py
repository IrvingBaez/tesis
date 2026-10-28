import argparse
import copy
import json
import logging
import os
import shutil
import time
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm

from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.third_party.avr_net.tools.write_rttms import main as write_rttms
from model.avd.score_avd import main as score_avd
from model.util import argparse_helper, get_path, save_data
from .feature_extraction import main as extract_features
from .tools.clustering_dataset import BlockShuffleSampler, ClusteringDataset
from .tools.episodic_dataset import EpisodicDataset, EpochSampler
from .tools.custom_collator import CustomCollator
from .tools.contrastive_loss import ContrastiveLoss

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.functional.classification import binary_f1_score, binary_precision, binary_recall

AHC_THRESHOLDS = [0.35, 0.375, 0.40, 0.425, 0.45, 0.475]


def empty_similarities(utterance_counts):
	"""One similarity matrix per video, with ones in the diagonal."""
	return {video_id: torch.eye(count) for video_id, count in utterance_counts.items()}


def store_scores(similarities, batch, scores):
	"""
	Writes the scores of a batch of pairs straight into their symmetric similarity matrices.
	Keeping one tuple per pair instead takes tens of GB on the full train set (~12.6M pairs).
	"""
	scores = scores.detach().flatten().float().cpu()
	index_a, index_b = batch['index_a'].flatten().cpu(), batch['index_b'].flatten().cpu()

	for video_id in dict.fromkeys(batch['video_id']):
		mask = torch.tensor([pair_video == video_id for pair_video in batch['video_id']])
		similarities[video_id][index_a[mask], index_b[mask]] = scores[mask]
		similarities[video_id][index_b[mask], index_a[mask]] = scores[mask]


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

		self.model = Attention_AVRNet(
			self.args.self_attention, self.args.cross_attention,
			dropout=self.args.self_attention_dropout, relation_init=self.args.relation_init
		)
		# self.model.freeze_relation()

		if 'fine_tunning' not in self.args:
			self.args.fine_tunning = False

		if self.args.fine_tunning:
			self.model.fine_tunning()

		self.cos = nn.CosineSimilarity()
		self.train_cos_stats = {'video': [], 'audio': [], 'target': []}
		self.val_cos_stats = {'video': [], 'audio': [], 'target': []}

		self.train_f1        = BinaryF1Score()
		self.train_recall    = BinaryRecall()
		self.train_precision = BinaryPrecision()
		self.val_f1          = BinaryF1Score()
		self.val_recall      = BinaryRecall()
		self.val_precision   = BinaryPrecision()

		self.eval_similarities = None


	def configure_optimizers(self):

		if self.args.optimizer == 'sgd':
			optimizer = torch.optim.SGD(
				self.parameters(),
				lr=self.args.learning_rate,
				weight_decay=self.args.weight_decay,
				momentum=self.args.momentum
			)
		elif self.args.optimizer == 'adam':
			optimizer = torch.optim.Adam(
				self.parameters(),
				lr=self.args.learning_rate,
				weight_decay=self.args.weight_decay
			)
		else:
			raise ValueError(f"optimizer must be 'sgd' or 'adam' not '{self.args.optimizer}'")

		lr_scheduler = StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

		return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


	def on_train_epoch_start(self):
		self.epoch_start_time = time.time()
		self.train_cos_stats = {'video': [], 'audio': [], 'target': []}

		if not self.args.fine_tunning and self.current_epoch >= self.args.frozen_epochs:
			self.model.unfreeze_relation()


	def training_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		scores = self.model(video, audio, task_full)

		loss = self.loss_fn(scores, target)

		accuracy = ((scores > 0.5) == target).float().mean()
		self.train_f1.update(scores, target)
		self.train_recall.update(scores, target)
		self.train_precision.update(scores, target)

		self.log("loss/train",      loss,                 on_step=False, on_epoch=True, sync_dist=True)
		self.log("acc/train",       accuracy,             on_step=False, on_epoch=True, sync_dist=True)
		self.log("f1_score/train",  self.train_f1,        on_step=False, on_epoch=True, sync_dist=True)
		self.log("recall/train",    self.train_recall,    on_step=False, on_epoch=True, sync_dist=True)
		self.log("precision/train", self.train_precision, on_step=False, on_epoch=True, sync_dist=True)

		return loss


	# The train DER is not computed here: it is computed once, for the best epoch, by score_best_epoch_on_train
	def on_train_epoch_end(self):
		# Validation runs before this hook, so its summary already exists
		summary_path = f'{self.logger.log_dir}/val_{self.current_epoch}/summary.json'
		if os.path.exists(summary_path):
			with open(summary_path, 'r') as file:
				summary = json.load(file)
			summary['epoch_seconds'] = time.time() - self.epoch_start_time
			with open(summary_path, 'w') as file:
				json.dump(summary, file, indent=2)


	def on_validation_epoch_start(self):
		self._start_eval_epoch()
		self.val_cos_stats = {'video': [], 'audio': [], 'target': []}


	def validation_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		scores = self.model(video, audio, task_full)

		loss = self.loss_fn(scores, target)

		accuracy = ((scores > 0.5) == target).float().mean()
		self.val_f1.update(scores, target)
		self.val_recall.update(scores, target)
		self.val_precision.update(scores, target)

		self.log("loss/val",        loss,               sync_dist=True)
		self.log("acc/val",         accuracy,           sync_dist=True)
		self.log("f1_score/val",    self.val_f1,        sync_dist=True)
		self.log("recall/val",      self.val_recall,    sync_dist=True)
		self.log("precision/val",   self.val_precision, sync_dist=True)

		self._store_eval_scores(batch, scores, target)

		return loss


	# Calculate DER as part of validation end
	def on_validation_epoch_end(self):
		similarity_data, scores, targets = self._gather_predictions(self.args.val_starts, self.args.val_ends)

		# Sanity check only scores a couple of batches, its results are not kept
		if self.trainer.sanity_checking:
			epoch_dir = f'{self.logger.log_dir}/sanity_check'
		else:
			epoch_dir = f'{self.logger.log_dir}/val_{self.current_epoch}'

		# Similarities and the RTTMs of every threshold are kept in {log_dir}/val_{epoch}
		results = self._score_ahc_sweep(
			similarity_data=similarity_data,
			similarities_path=f'{epoch_dir}/similarities.pth',
			base_sys_path=f'{epoch_dir}/rttms',
			data_type='val',
			ref_path=self.args.val_ref_out,
		)
		threshold = min(results, key=lambda t: results[t]['der'])
		self._log_threshold(results, 'val', threshold)

		if self.trainer.sanity_checking:
			shutil.rmtree(epoch_dir)
			return

		self._write_summary(epoch_dir, results, scores, targets, epoch=self.current_epoch, threshold=threshold, **results[threshold])


	def on_test_epoch_start(self):
		self._start_eval_epoch()


	def test_step(self, batch, batch_idx):
		video, audio, task_full, target = batch['video'], batch['audio'], batch['task_full'], batch['target']
		scores = self.model(video, audio, task_full)

		loss = self.loss_fn(scores, target)

		accuracy = ((scores > 0.5) == target).float().mean()
		self.val_f1.update(scores, target)
		self.val_recall.update(scores, target)
		self.val_precision.update(scores, target)

		self.log("loss/test",       loss,               sync_dist=True)
		self.log("acc/test",        accuracy,           sync_dist=True)
		self.log("f1_score/test",   self.val_f1,        sync_dist=True)
		self.log("recall/test",     self.val_recall,    sync_dist=True)
		self.log("precision/test",  self.val_precision, sync_dist=True)

		self._store_eval_scores(batch, scores, target)

		return loss


	# Calculate DER as part of test end
	def on_test_epoch_end(self):
		similarity_data, scores, targets = self._gather_predictions(self.args.val_starts, self.args.val_ends)
		test_dir = f'{self.logger.log_dir}/test'

		# Similarities and the RTTMs of every threshold are kept in {log_dir}/test
		results = self._score_ahc_sweep(
			similarity_data=similarity_data,
			similarities_path=f'{test_dir}/similarities.pth',
			base_sys_path=f'{test_dir}/rttms',
			data_type='test',
		)

		# The threshold is never chosen on test: it is the one of the best validation epoch of this run
		best_val = best_val_summary(self.logger.log_dir)
		if best_val is not None:
			threshold, source = best_val['threshold'], f"validation epoch {best_val['epoch']}"
		else:
			threshold, source = self.args.ahc_threshold, 'args.ahc_threshold (the run has no validation summaries)'
		print(f'\nTest DER reported with AHC threshold {threshold}, from {source}')

		self._log_threshold(results, 'test', threshold)
		self._write_summary(test_dir, results, scores, targets, threshold=threshold, threshold_source=source, **results[threshold])


	def _start_eval_epoch(self):
		self.eval_similarities = empty_similarities(self.args.val_utterance_counts)
		self.eval_scores, self.eval_targets = [], []


	def _store_eval_scores(self, batch, scores, target):
		store_scores(self.eval_similarities, batch, scores)
		self.eval_scores.append(scores.detach().flatten().cpu())
		self.eval_targets.append(target.detach().flatten().cpu())


	def _gather_predictions(self, starts, ends):
		"""Returns the similarity matrices filled during evaluation, plus every pair score and label."""
		similarity_data = {'similarities': self.eval_similarities, 'starts': starts, 'ends': ends}
		scores, targets = torch.cat(self.eval_scores), torch.cat(self.eval_targets)
		self.eval_similarities, self.eval_scores, self.eval_targets = None, [], []

		return similarity_data, scores, targets


	def _write_summary(self, folder, results, scores, targets, **extra):
		"""Writes {folder}/summary.json with the DER of every AHC threshold and the pair classification metrics."""
		summary = {
			**extra,
			'f1':			binary_f1_score(scores, targets.int()).item(),
			'precision':	binary_precision(scores, targets.int()).item(),
			'recall':		binary_recall(scores, targets.int()).item(),
			'results':		{f'{threshold}': metrics for threshold, metrics in results.items()},
		}

		with open(f'{folder}/summary.json', 'w') as file:
			json.dump(summary, file, indent=2)


	def _collect_similarities(self, dataset, desc='Collecting predictions'):
		loader = DataLoader(
			dataset, batch_size=512, shuffle=False,
			num_workers=0, collate_fn=CustomCollator(),
		)
		similarities = empty_similarities(dataset.utterance_counts)
		self.model.eval()

		with torch.no_grad():
			for batch in tqdm(loader, desc=desc):
				video     = batch['video'].to(self.device)
				audio     = batch['audio'].to(self.device)
				task_full = batch['task_full'].to(self.device)
				scores    = self.model(video, audio, task_full)
				store_scores(similarities, batch, scores)

		self.model.train()
		return similarities


	def _score_ahc_sweep(self, similarity_data, similarities_path, base_sys_path, data_type, ref_path=None,
		thresholds=AHC_THRESHOLDS, log=True):
		"""
		Clusters with every threshold in thresholds and scores the resulting RTTMs, which are left in {base_sys_path}_{tag}.
		Returns {threshold: {'der', 'miss', 'fa', 'confusion'}}, all in percent. log=False outside the Lightning loops.
		"""
		os.makedirs(Path(similarities_path).parent, exist_ok=True)
		save_data(similarity_data, similarities_path, verbose=False, override=True)
		results = {}

		for threshold in tqdm(thresholds, desc=f'AHC sweep [{data_type}, epoch {self.current_epoch}]'):
			tag = int(threshold * 100)
			sys_path = f'{base_sys_path}_{tag}'
			write_rttms(similarities_path=similarities_path, sys_path=sys_path, data_type=data_type, ahc_threshold=threshold)

			scores_out		= f'{sys_path}/{data_type}.out'
			output_path		= f'{sys_path}/{data_type}_scores.out'
			breakdown_path	= f'{sys_path}/{data_type}_breakdown.json'
			if ref_path is not None:
				score_avd(data_type=data_type, sys_path=scores_out, output_path=output_path, ref_path=ref_path, breakdown_path=breakdown_path)
			else:
				score_avd(data_type=data_type, sys_path=scores_out, output_path=output_path, breakdown_path=breakdown_path)

			with open(output_path, 'r') as file:
				content = file.read()
			der = float(content.strip().split('\n')[-1].split()[3])

			with open(breakdown_path, 'r') as file:
				breakdown = json.load(file)

			results[threshold] = {'der': der, 'miss': breakdown['miss'], 'fa': breakdown['fa'], 'confusion': breakdown['confusion']}
			if log: self.log(f"der/{data_type}_{tag}", der, sync_dist=True)

		return results


	def _log_threshold(self, results, data_type, threshold):
		self.log(f"der/{data_type}",			results[threshold]['der'],			sync_dist=True)
		self.log(f"der_miss/{data_type}",		results[threshold]['miss'],			sync_dist=True)
		self.log(f"der_fa/{data_type}",			results[threshold]['fa'],			sync_dist=True)
		self.log(f"der_confusion/{data_type}",	results[threshold]['confusion'],	sync_dist=True)
		self.log(f"ahc_threshold/{data_type}",	threshold,							sync_dist=True)


	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		predictions = []
		scores = self.model(batch['video'], batch['audio'], batch['task_full'])

		for video_id, index_a, index_b, score in zip(batch['video_id'], batch['index_a'], batch['index_b'], scores):
			predictions.append((video_id, index_a, index_b, score.cpu()))

		return predictions



CHECKPOINT_DIR = 'model/third_party/avr_net/checkpoints/'


def run_dir(run_name):
	return os.path.join(CHECKPOINT_DIR, 'lightning_logs', run_name)


def epoch_summaries(folder):
	"""Validation summary.json of every epoch in a run folder, sorted by epoch."""
	summaries = []
	for path in glob(os.path.join(folder, 'val_*', 'summary.json')):
		with open(path, 'r') as file:
			summaries.append(json.load(file))

	return sorted(summaries, key=lambda summary: summary['epoch'])


def best_val_summary(folder):
	"""Validation epoch with the lowest DER (the first one on ties), the same criterion as ModelCheckpoint."""
	summaries = epoch_summaries(folder)

	return min(summaries, key=lambda summary: summary['der']) if summaries else None


def score_best_epoch_on_train(model, dataset, logger):
	"""
	DER over every train pair, with the weights and the AHC threshold of the best validation epoch.
	It runs once, after the last epoch, because one pass over the 12.6 M train pairs takes ~1 h 40 min.
	Results are left in {log_dir}/train/summary.json and logged as der/train at the step of that epoch.
	"""
	best_val = best_val_summary(logger.log_dir)
	epoch, threshold = best_val['epoch'], best_val['threshold']
	checkpoint = glob(os.path.join(logger.log_dir, 'checkpoints', f'epoch={epoch}-*.ckpt'))[0]
	print(f'\nTrain DER with the weights and AHC threshold ({threshold}) of validation epoch {epoch}: {checkpoint}')

	model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'])
	model.to('cuda' if torch.cuda.is_available() else 'cpu')
	similarities = model._collect_similarities(dataset, desc=f'Collecting predictions [train, epoch {epoch}]')

	train_dir = os.path.join(logger.log_dir, 'train')
	similarity_data = {'similarities': similarities, 'starts': model.args.train_starts, 'ends': model.args.train_ends}
	results = model._score_ahc_sweep(
		similarity_data=similarity_data,
		similarities_path=f'{train_dir}/similarities.pth',
		base_sys_path=f'{train_dir}/rttms',
		data_type='train',
		ref_path=model.args.train_ref_out,
		thresholds=[threshold],
		log=False,
	)
	os.remove(f'{train_dir}/similarities.pth')  # ~100 MB per run, the RTTMs are kept

	with open(f'{train_dir}/summary.json', 'w') as file:
		json.dump({'epoch': epoch, 'threshold': threshold, **results[threshold]}, file, indent=2)

	logger.experiment.add_scalar('der/train', results[threshold]['der'], epoch)
	logger.experiment.flush()
	print(f"Train DER: {results[threshold]['der']}")


def initialize_arguments(**kwargs):
	# Resume: restore the saved config of the run and continue from its last checkpoint
	if kwargs.get('resume'):
		assert kwargs.get('run_name'), 'Cannot resume without a run_name'
		config_path = os.path.join(run_dir(kwargs['run_name']), 'config.json')
		print(f'\nLoading config from: {config_path}')
		with open(config_path, 'r') as f:
			saved_config = json.load(f)
		saved_config.pop('environment', None)
		kwargs = {
			**saved_config,
			'run_name':		kwargs['run_name'],
			'resume':		True,
			'checkpoint':	os.path.join(run_dir(kwargs['run_name']), 'checkpoints', 'last.ckpt'),
			'not_empty':	True,
		}

	parser = argparse.ArgumentParser(description = "Attention AVR-Net training")

	# TRAINING CONFIGURATION
	parser.add_argument('--learning_rate',			type=float,	default=0.001, 			help='Training base learning rate')
	parser.add_argument('--momentum',				type=float, default=0.0,			help='Training momentum for SDG optimizer. Ignored if Adam optimizer is selected.')
	parser.add_argument('--weight_decay',			type=float, default=0.0001,			help='Training weight decay for SDG optimizer')
	parser.add_argument('--step_size',				type=int, 	default=5,				help='Training stepsize for StepLR scheduler')
	parser.add_argument('--gamma',					type=float, default=0.5,			help='Training gamma for StepLR scheduler')
	parser.add_argument('--video_proportion', 		type=float, default=1.0,			help='Percentage of available videos to use in training')
	parser.add_argument('--val_video_proportion', 	type=float, default=1.0,			help='Percentage of available videos to use in validation')
	parser.add_argument('--epochs', 				type=int, 	default=10, 			help='Fixed number of training epochs')
	parser.add_argument('--seed', 					type=int, 	default=42, 			help='Seed for python, numpy and torch RNGs')
	parser.add_argument('--frozen_epochs',			type=int, 	default=0, 				help='Epochs to train without updating relation network weights')
	parser.add_argument('--self_attention', 		type=str, 	default='',				help='Self attention method to marge available frame features')
	parser.add_argument('--self_attention_dropout', type=float, default=0.1, 			help='Dropout used in self-attention transformer')
	parser.add_argument('--cross_attention', 		type=str, 	default='', 			help='Cross attention method to marge frame and audio features')
	parser.add_argument('--loss_fn', 				type=str, 	default='', 			help='Loss function to use during training')
	parser.add_argument('--pos_margin', 			type=float, default=0.0, 			help='Positive margin for contrastive loss')
	parser.add_argument('--neg_margin', 			type=float, default=1.0, 			help='Negative margin for contrastive loss')
	parser.add_argument('--ahc_threshold', 			type=float, default=0.3, 			help='Threshold distance for the AHC algorithm')
	parser.add_argument('--optimizer', 				type=str, 	default='', 			help='Optimizer to use during training')
	parser.add_argument('--task', 					type=str, 	default='train', 		help='Execution mode, either train or val')
	parser.add_argument('--disable_pb', 			action='store_true', 				help='If true, hides progress bars')
	# DATA CONFIGURATION
	parser.add_argument('--max_frames', 			type=int, 	default=1,				help='How many frames to use in self-attention')
	parser.add_argument('--db_video_mode', 			type=str, 	default='pick_first',	help='Selection mode for video frames in the dataset')
	parser.add_argument('--frame_selection', 		type=str, 	default='first',		help="Which max_frames of each utterance are used: 'first' or 'center'")
	parser.add_argument('--aligned', 				action='store_true', 				help='Wether or not to use aligned frames')
	parser.add_argument('--balanced',				action='store_true', 				help='Balance positives and negatives examples in training data.')
	parser.add_argument('--training', 				type=str, 	default='pairs',		help="'pairs': every pair (or balanced pairs) of each video. 'episodic': episodes of the original AVR-Net")
	parser.add_argument('--shuffle_block_videos',	type=int, 	default=0,				help='0: preload the train features and shuffle every pair. N: shuffle within blocks of N videos (features that do not fit in memory)')
	# MODEL CONFIGURATION
	parser.add_argument('--relation_init', 			type=str,		default=None, 		help='Path of pretrained relation layer weights. If not given, trains from scratch.')
	parser.add_argument('--checkpoint', 			type=str,		default=None, 		help='Path of checkpoint to load weights from (or to evaluate).')
	parser.add_argument('--run_name', 				type=str,		default=None, 		help='Run identifier, logs are written to lightning_logs/<run_name>.')
	parser.add_argument('--resume', 				action='store_true', 				help='Continue run_name from its last checkpoint.')
	parser.add_argument('--fine_tunning',			action='store_true', 				help='Freezes all but the last relation layer.')

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
	assert Path(args.val_dataset_config['waves_path']).exists(), 	f'Data path {args.val_dataset_config['waves_path']} does not exist.'
	assert Path(args.val_dataset_config['rttms_path']).exists(), 	f'Data path {args.val_dataset_config['rttms_path']} does not exist.'
	assert Path(args.val_dataset_config['labs_path']).exists(), 	f'Data path {args.val_dataset_config['labs_path']} does not exist.'
	assert Path(args.val_dataset_config['frames_path']).exists(), 	f'Data path {args.val_dataset_config['frames_path']} does not exist.'

	args.sys_path 			= 'model/third_party/avr_net/features'
	args.checkpoint_dir = CHECKPOINT_DIR

	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	return args


def save_experiment_config(args, log_dir):
	_CONFIG_PARAMS = [
		'run_name', 'seed', 'relation_init',
		'video_proportion', 'val_video_proportion', 'aligned', 'balanced', 'training', 'shuffle_block_videos',
		'max_frames', 'frame_selection', 'db_video_mode', 'checkpoint', 'task',
		'self_attention', 'self_attention_dropout', 'cross_attention', 'fine_tunning',
		'loss_fn', 'pos_margin', 'neg_margin', 'ahc_threshold',
		'optimizer', 'learning_rate', 'momentum', 'weight_decay',
		'step_size', 'gamma', 'epochs', 'frozen_epochs', 'disable_pb',
	]

	config = {k: getattr(args, k) for k in _CONFIG_PARAMS if hasattr(args, k)}
	config['environment'] = {
		'gpu':			torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
		'torch':		torch.__version__,
		'cuda':			torch.version.cuda,
		'lightning':	pl.__version__,
	}
	os.makedirs(log_dir, exist_ok=True)
	with open(os.path.join(log_dir, 'config.json'), 'w') as f:
		json.dump(config, f, indent=2)


def generate_filtered_ref(video_ids, data_type, output_path):
	full_ref_path = get_path('avd_path', avd_detector='ground_truth') + f'/{data_type}.out'
	selected_ids = set(video_ids)

	with open(full_ref_path) as f:
		filtered = [l for l in f if l.strip() and l.strip().split('/')[-1].replace('.rttm', '') in selected_ids]

	with open(output_path, 'w') as f:
		f.writelines(filtered)


def train(args):
	torch.set_float32_matmul_precision('high')
	pl.seed_everything(args.seed, workers=True)

	if args.task == 'train' and args.run_name and not args.resume:
		assert not Path(run_dir(args.run_name)).exists(), \
			f'Run {args.run_name} already exists in {run_dir(args.run_name)}. Resume it or delete it first.'

	train_loader, eval_loader = create_dataset(args)

	args.val_utterance_counts = eval_loader.dataset.utterance_counts
	args.val_starts = eval_loader.dataset.starts()
	args.val_ends = eval_loader.dataset.ends()

	if args.balanced:
		train_eval_dataset = ClusteringDataset(
			args.train_features_path, args.disable_pb,
			video_proportion=args.video_proportion, balanced=False,
		)
	else:
		train_eval_dataset = None

	if train_eval_dataset is not None:	train_der_dataset = train_eval_dataset
	elif args.training == 'episodic':	train_der_dataset = train_loader.dataset.pairs
	else:								train_der_dataset = train_loader.dataset
	args.train_utterance_counts = train_der_dataset.utterance_counts
	args.train_starts           = train_der_dataset.starts()
	args.train_ends             = train_der_dataset.ends()

	args.train_ref_out = f'{args.train_features_path}/ref_train.out'
	generate_filtered_ref(list(train_der_dataset.utterance_counts.keys()), 'train', args.train_ref_out)

	args.val_ref_out = f'{args.val_features_path}/ref_val.out'
	generate_filtered_ref(list(eval_loader.dataset.utterance_counts.keys()), 'val', args.val_ref_out)

	# Full resume: restore weights + optimizer + epoch counter, write to same folder.
	# Partial load: load only weights (e.g. fine-tuning from a different run).
	full_resume = args.checkpoint and args.resume
	ckpt_path = args.checkpoint if full_resume else None

	model = Lightning_Attention_AVRNet(clean_up_args(args))
	if args.checkpoint and not full_resume:
		print(f'\nLoading checkpoint weights from: {args.checkpoint}')
		state_dict = torch.load(args.checkpoint)['state_dict']
		model.load_state_dict(state_dict)
	elif full_resume:
		print(f'\nResuming training from: {args.checkpoint} ({args.run_name})')
	elif args.relation_init:
		print(f'\nLoading relation layer weights from: {args.relation_init}')
	else:
		print('\nTraining relation layer from scratch')

	logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

	logger = TensorBoardLogger(save_dir=args.checkpoint_dir, version=args.run_name)

	trainer = pl.Trainer(
		accelerator="gpu", devices=1, strategy="auto", # auto ddp_find_unused_parameters_true ddp
		logger=logger,
		min_epochs=args.epochs,
		max_epochs=args.epochs,
		callbacks=[
			ModelCheckpoint(monitor="der/val", mode="min", save_last=True),
		],
	)

	# On resume, keep the original config of the run. Evaluating val or test does not replace the training config
	if args.task == 'train' and not args.resume:
		save_experiment_config(args, trainer.logger.log_dir)
	print()

	if args.task == 'train':
		trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=eval_loader, ckpt_path=ckpt_path)

		# Lightning returns normally after Ctrl+C (trainer.interrupted), so only a finished fit marks the run as done.
		# If the train DER is interrupted, the run stays resumable: fit returns at once and the train DER runs again
		if trainer.state.finished and args.run_name:
			score_best_epoch_on_train(model, train_der_dataset, trainer.logger)
			with open(os.path.join(run_dir(args.run_name), 'done.json'), 'w') as file:
				json.dump({'epochs': trainer.current_epoch, 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S')}, file, indent=2)
	elif args.task == 'val':
		assert args.checkpoint is not None, 'Cannot eval without a given checkpoint'
		trainer.validate(model=model, dataloaders=eval_loader)
	elif args.task == 'test':
		assert args.checkpoint is not None, 'Cannot test without a given checkpoint'
		trainer.test(model=model, dataloaders=eval_loader)

	return trainer


def clean_up_args(args):
		allowed_args = ['loss_fn', 'val_starts', 'val_ends', 'val_utterance_counts', 'val_ref_out', 'train_starts', 'train_ends', 'train_utterance_counts', 'train_ref_out', 'self_attention', 'cross_attention', 'self_attention_dropout', 'relation_init', 'fine_tunning', 'learning_rate', 'weight_decay', 'momentum', 'optimizer', 'step_size', 'gamma', 'frozen_epochs', 'pos_margin', 'neg_margin', 'ahc_threshold']
		current_args = [key for key in args.__dict__.keys()]

		args_copy = copy.deepcopy(args)

		for key in current_args:
			if key not in allowed_args:
				args_copy.__dict__.pop(key)

		not_empty = ['val_starts', 'val_ends', 'val_utterance_counts', 'val_ref_out', 'train_starts', 'train_ends', 'train_utterance_counts', 'train_ref_out']

		for element in not_empty:
			if element not in args_copy.__dict__.keys():
				args_copy.__dict__[element] = None

		return args_copy


def create_dataset(args):
	args.train_features_path, args.val_features_path, args.test_features_path = extract_features(
		sys_path=args.sys_path,
		aligned=args.aligned,
		max_frames=args.max_frames,
		frame_selection=args.frame_selection,
		db_video_mode=args.db_video_mode,
		disable_pb=args.disable_pb
	)

	train_loader = load_data(args, mode='train', workers=11, batch_size=64)
	eval_mode = 'test' if args.task == 'test' else 'val'
	eval_loader = load_data(args, mode=eval_mode, workers=2, batch_size=512)

	return train_loader, eval_loader


def load_data(args, mode, workers, batch_size=256):
	sampler = None
	if mode=='train' and args.training == 'episodic':
		# Only the pairs differ from balanced training: same number per epoch, same batches (DECISIONES.md R3.2)
		pairs = ClusteringDataset(args.train_features_path, args.disable_pb, video_proportion=args.video_proportion, preload=True)
		dataset = EpisodicDataset(pairs, args.epochs, args.seed, disable_pb=args.disable_pb)
		sampler = EpochSampler(dataset, args.seed)
		print(f'Episodic training: {dataset.episodes_per_epoch} episodes, {dataset.pairs_per_epoch} pairs per epoch')
	elif mode=='train' and args.shuffle_block_videos:
		dataset = ClusteringDataset(args.train_features_path, args.disable_pb, video_proportion=args.video_proportion, balanced=args.balanced, cache_size=args.shuffle_block_videos)
		sampler = BlockShuffleSampler(dataset, args.shuffle_block_videos, args.seed)
	elif mode=='train':
		dataset = ClusteringDataset(args.train_features_path, args.disable_pb, video_proportion=args.video_proportion, balanced=args.balanced, preload=True)
	elif mode=='val':
		dataset = ClusteringDataset(args.val_features_path, args.disable_pb, video_proportion=args.val_video_proportion)
	elif mode=='test':
		print('Ignoring val_video_proportion, testing with proportion 1.0')
		dataset = ClusteringDataset(args.test_features_path, args.disable_pb, video_proportion=1.0)
	else:
		raise ValueError(f"Dataset mode '{mode}' not recognized")

	# Train pairs (or episodes) are shuffled every epoch (order fixed by the seed), within blocks of videos when
	# shuffle_block_videos is set. Evaluation keeps the order of the videos
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=(mode == 'train' and sampler is None),
		sampler=sampler,
		num_workers=workers,
		persistent_workers=True,
		pin_memory=True,
		drop_last=False,
		collate_fn=CustomCollator()
	)

	return dataloader


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	return train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
