import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from model.util import argparse_helper, save_data, get_path
from .tools.feature_extractor import FeatureExtractor
from .tools.dataset import CustomDataset
from .tools.custom_collator import CustomCollator

def extract_all_features(args):
	os.makedirs(f'{args.sys_path}', exist_ok=True)

	with torch.no_grad():
		extract_features(args, mode='train')
		extract_features(args, mode='val')


def extract_features(args, mode):
	feature_extractor = FeatureExtractor(args.db_video_mode)
	feature_extractor = nn.DataParallel(feature_extractor)
	feature_extractor.to(args.device)

	if mode == 'train':
		config = args.train_dataset_config
	else:
		config = args.val_dataset_config

	output_path = f'{args.features_path}/{mode}'
	os.makedirs(output_path, exist_ok=True)

	video_ids = config['video_ids']
	for video_id in tqdm(video_ids, desc=f'Extracting features for {mode}', disable=args.disable_pb):
		video_feats_path = f'{output_path}/{video_id}.pckl'
		if os.path.exists(video_feats_path): continue

		config['video_ids'] = [video_id]

		dataset = CustomDataset(config, training=True, disable_pb=args.disable_pb)
		dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=CustomCollator())

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
		save_data(features, video_feats_path)


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
				value = torch.cat((features[key], value))
				features[key] = value
				del value
			elif isinstance(value, dict):
				features[key] = merge_features([features[key], value])

	for key, value in features.items():
		if isinstance(value, torch.Tensor):
			features[key] = value.cpu()

	return features


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "AVA-AVD utterance feature extraction.")

	parser.add_argument('--sys_path', 				type=str, 	help='Root path to save features. Depreciated.', default='')
	parser.add_argument('--aligned', 					action='store_true', help='Wether or not to use alined frames')
	parser.add_argument('--max_frames', 			type=int, help='How many frames to use in self-attention')
	parser.add_argument('--db_video_mode', 		type=str, help='Selection mode for video frames in the dataset')
	parser.add_argument('--disable_pb', 			action='store_true', help='If true, hides progress bars')

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

	args.sys_path = 'model/third_party/avr_net/features'
	args.features_path = f'{args.sys_path}/{args.max_frames}_frames'
	if args.aligned: args.features_path += '_aligned'
	args.features_path += f'_{args.db_video_mode}'

	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	extract_all_features(args)

	return args.features_path + '/train', args.features_path + '/val'


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
