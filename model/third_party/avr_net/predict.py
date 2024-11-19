import torch, argparse, os
import torch.nn as nn
from shutil import rmtree
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.third_party.avr_net.tools.custom_collator import CustomCollator
from model.third_party.avr_net.tools.dataset import CustomDataset
from model.third_party.avr_net.tools.clustering_dataset import ClusteringDataset
from model.third_party.avr_net.tools.feature_extractor import FeatureExtractor
from model.third_party.avr_net.attention_avr_net import Attention_AVRNet
from model.third_party.avr_net.tools.write_rttms import main as write_rttms

from model.util import argparse_helper, save_data, check_system_usage, show_similarities


def predict(args):
	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(f'{args.sys_path}')

	features = extract_features(args)
	save_data(features, args.features_path)
	similarities = compute_similarity(args)

	save_data(similarities, args.similarities_path)
	show_similarities('similarities_testing', similarities['similarities'])

	write_rttms(similarities_path=args.similarities_path, sys_path=args.sys_path, data_type=args.data_type)


def extract_features(args):
	feature_extractor = FeatureExtractor()
	feature_extractor = nn.DataParallel(feature_extractor)
	feature_extractor.to(args.device)

	datset_config = {
		'video_ids':	 args.video_ids,
		'waves_path':	 args.waves_path,
		'rttms_path':	 args.rttms_path,
		'labs_path':	 args.labs_path,
		'frames_path': args.frames_path
	}

	dataset = CustomDataset(datset_config, training=True)
	dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())
	dataloader = tqdm(dataloader, desc='Extracting features')

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


def compute_similarity(args):
	similarities = {}
	model = load_model(args)

	dataset = ClusteringDataset(args.features_path)
	dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())

	for video_id in args.video_ids:
		similarities[video_id] = torch.diag_embed(torch.ones([dataset.count_utterances(video_id)]))

	with torch.no_grad():
		for batch in tqdm(dataloader, desc='Clustering features'):
			scores = model(batch['video'], batch['audio'], batch['task_full'])

			for video_id, index_a, index_b, score in zip(batch['video_id'], batch['index_a'], batch['index_b'], scores):
				similarity = similarities[video_id]
				similarity[index_a, index_b] = similarity[index_b, index_a] = score.cpu()

	return {'similarities': similarities, 'starts': dataset.start, 'ends': dataset.end}


def load_model(args):
	model = Attention_AVRNet()
	model= nn.DataParallel(model)

	model.to(args.device)
	if args.checkpoint and os.path.isfile(args.checkpoint):
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])

	model.eval()

	return model


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--data_type',		type=str,	help='Type of data being processed, test, val or train')
	parser.add_argument('--video_ids',		type=str,	help='Video ids separated by commas')
	parser.add_argument('--videos_path',	type=str,	help='Path to the videos to work with')
	parser.add_argument('--waves_path',		type=str,	help='Path to the waves, already denoised')
	parser.add_argument('--labs_path',		type=str,	help='Path to the lab files with voice activity detection info')
	parser.add_argument('--rttms_path',		type=str,	help='Path to the lab files with voice activity detection info')
	parser.add_argument('--frames_path',	type=str,	help='Path to the face frames already cropped and aligned')
	parser.add_argument('--tracks_path',	type=str,	help='Path to the csv files containing the active speaker detection info')
	parser.add_argument('--sys_path',			type=str,	help='Path to the folder where to save all the system outputs')

	# MODEL CONFIGURATION
	parser.add_argument('--relation_layer', type=str, help='Type of relation to use', default='original')
	parser.add_argument('--checkpoint', 		type=str,	help='Path of checkpoint to load and eval', default=None)

	args = argparse_helper(parser, **kwargs)

	args.video_ids = args.video_ids.split(',')
	args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	args.similarities_path = f'{args.sys_path}/similarity_matrix.pckl'
	args.features_path = f'{args.sys_path}/features.pckl'

	# if not args.checkpoint:
	# 	args.checkpoint = 'model/third_party/avr_net/weights/best_relation.ckpt'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	predict(args)


if __name__ == '__main__':
	args = initialize_arguments()
	predict(args)
