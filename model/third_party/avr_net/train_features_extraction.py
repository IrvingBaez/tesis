import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .tools.clustering_dataset import ClusteringDataset
from tqdm.auto import tqdm
from pathlib import Path
import random

from model.util import argparse_helper, save_data, get_path, check_system_usage, file_size
from model.third_party.avr_net.tools.feature_extractor import FeatureExtractor
from model.third_party.avr_net.tools.custom_collator import CustomCollator
from model.third_party.avr_net.tools.dataset import CustomDataset


def load_features(args, video_id):
	os.makedirs(f'{args.sys_path}', exist_ok=True)

	with torch.no_grad():
		if not args.disable_pb: print('======================== Extracting Features ========================')
		train_features = extract_features(args, mode='train', video_ids=[video_id])
		save_data(train_features, args.train_features_path, override=True)
		del train_features
		if not args.disable_pb: print('======================== Features Extracted ========================')

	if not args.disable_pb: print('Features file size:\t', end='')
	if not args.disable_pb: print(file_size(args.train_features_path))


def extract_features(args, mode, video_ids):
	feature_extractor = FeatureExtractor()
	feature_extractor = nn.DataParallel(feature_extractor)
	feature_extractor.to(args.device)

	if mode == 'train':
		config = args.train_dataset_config
	else:
		config = args.val_dataset_config

	dataset = CustomDataset(config, training=True, disable_pb=args.disable_pb, video_ids=video_ids)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())
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


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "AVA-AVD data exploration")

	parser.add_argument('--aligned', 		action='store_true', help='Wether or not to use alined frames')
	parser.add_argument('--disable_pb', action='store_true', help='If true, hides progress bars')

	args = argparse_helper(parser, **kwargs)

	with open(f'dataset/split/train.list', 'r') as file:
		args.video_ids = file.read().split('\n')

	args.train_dataset_config = {
		'video_ids':	 	args.video_ids,
		'waves_path':	 	get_path('waves_path', denoiser='dihard18'),
		'rttms_path':	 	get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':	 	get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path': 	get_path('asd_path', asd_detector='ground_truth') + ('/aligned_tracklets' if args.aligned else '/tracklets'),
		'max_frames':		10
	}

	with open(f'dataset/split/val.list', 'r') as file:
		val_video_ids = file.read().split('\n')

	args.val_dataset_config = {
		'video_ids':	 	val_video_ids,
		'waves_path':	 	get_path('waves_path', denoiser='dihard18'),
		'rttms_path':	 	get_path('avd_path', avd_detector='ground_truth') + '/predictions',
		'labs_path':	 	get_path('vad_path', vad_detector='ground_truth') + '/predictions',
		'frames_path': 	get_path('asd_path', asd_detector='ground_truth') + ('/aligned_tracklets' if args.aligned else '/tracklets'),
		'max_frames':		1
	}

	assert Path(args.train_dataset_config['waves_path']).exists()
	assert Path(args.train_dataset_config['frames_path']).exists()
	assert Path(args.val_dataset_config['waves_path']).exists()
	assert Path(args.val_dataset_config['frames_path']).exists()


	args.sys_path 						= '/export/fs06/ibaez6/features'
	args.train_features_path 	= f'{args.sys_path}/train_features.pckl'
	args.val_features_path 		= f'{args.sys_path}/val_features.pckl'
	args.intermediate_path 		= f'{args.sys_path}/intermediate.pckl'
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	return args


def extract_video_features(args, video_id):
	save_path = f'{args.sys_path}/{video_id}.pckl'
	if os.path.exists(save_path): return

	load_features(args, video_id)

	dataset = ClusteringDataset(args.train_features_path, disable_pb=args.disable_pb)

	tally = {
		'target_0': {
			'task_0': [],
			'task_1': [],
			'task_2': [],
			'task_3':	[]
		},
		'target_1': {
			'task_0': [],
			'task_1': [],
			'task_2': [],
			'task_3':	[]
		},
	}

	for i in range(len(dataset)):
		sample = dataset[i]

		target = f'target_{int(sample['target'].item())}'

		task_a, _ = sample['task_full']
		task = f'task_{task_a}'

		tally[target][task].append(i)

	all_values = [len(index_list) for index_list in tally['target_0'].values()] +	[len(index_list) for index_list in tally['target_1'].values()]
	min_count = min(all_values)

	# 2000 is a good number for full training
	min_count = min(min_count, 125)
	# print(min_count)

	desired_indices = random.sample(tally['target_0']['task_0'], min_count)
	desired_indices += random.sample(tally['target_0']['task_1'], min_count)
	desired_indices += random.sample(tally['target_0']['task_2'], min_count)
	desired_indices += random.sample(tally['target_0']['task_3'], min_count)
	desired_indices += random.sample(tally['target_1']['task_0'], min_count)
	desired_indices += random.sample(tally['target_1']['task_1'], min_count)
	desired_indices += random.sample(tally['target_1']['task_2'], min_count)
	desired_indices += random.sample(tally['target_1']['task_3'], min_count)

	dataset_test = [dataset[i] for i in desired_indices]

	save_data(dataset_test, save_path, verbose=False)
	print(f'\n{video_id}: {min_count * 8}/{sum(all_values)} ({file_size(save_path)})', flush=True)


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)

	for video_id in tqdm(args.train_dataset_config['video_ids'], desc='Extracting features from individual videos'):
		extract_video_features(args, video_id)
