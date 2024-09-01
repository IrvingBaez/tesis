import torch, argparse, pickle, os
import numpy as np
import torch.nn as nn

from model.third_party.avr_net.tools.predict_collator import PredictCollator
from model.third_party.avr_net.tools.ahc_cluster import AHC_Cluster
from model.third_party.avr_net.tools.dataset import CustomDataset
from model.third_party.avr_net.avr_net import AVRNET
from model.util import argparse_helper
from shutil import rmtree
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from glob import glob


CONFIG = {
	'audio': {
		'layers': [3, 4, 6, 3],
		'num_filters': [32, 64, 128, 256],
		'encoder_type': 'ASP',
		'n_mels': 64,
		'log_input': True,
		'fix_layers': 'all',
		'init_weight': 'model/third_party/avr_net/weights/backbone_audio.pth'
	},
	'video': {
		'layers': [3, 4, 14, 3],
		'fix_layers': 'all',
		'num_features': 512,
		'inplanes': 64,
		'init_weight': 'model/third_party/avr_net/weights/backbone_faces.pth'
	},
	'relation': {
		'dropout': 0,
		'num_way': 20,
		'layers': [8, 6],
		'num_shot': 1,
		'num_filters': [256, 64]
	},
	'checkpoint': 'model/third_party/avr_net/weights/best_0.14_20.66.ckpt'
}


def predict(args):
	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(args.sys_path)
	os.makedirs(f'{args.sys_path}/features')
	os.makedirs(f'{args.sys_path}/predictions')

	model, device = load_model()
	model.eval()

	with torch.no_grad():
		extract_features(model, device, args)
		cluster_features(model, args)

	rmtree(f'{args.sys_path}/features')


def load_model():
	# INSTANTIATE MODEL
	model = AVRNET(CONFIG)
	model.build()

	# LOAD MODEL TO GPU
	if torch.cuda.is_available():
		device = torch.device("cuda")
		# torch.cuda.set_device(0)
	else:
		device = torch.device("cpu")

	model= nn.DataParallel(model)
	model.to(device)

	return model, device


def extract_features(model, device, args):
	# LOAD DATA
	dataset = CustomDataset(args)
	dataset.load_dataset()

	# Set shuffle=true for training: test should be 17199 items, is 17063
	dataloader = DataLoader(dataset, batch_size=255, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, collate_fn=PredictCollator())

	# INSTANTIATE OPTIMIZER
	# parameters = list(model.parameters())
	# optimizer = torch.optim.Adam(parameters, lr=0.0005, weight_decay=0.0001)
	# scaler = torch.cuda.amp.GradScaler(enabled=False)

	dataloader = tqdm(dataloader, desc='Extracting features')

	n = 0
	for batch in dataloader:
		batch['frames'] = batch['frames'].to(device)
		batch['audio'] = batch['audio'].to(device)

		model_output = model(batch, exec='extraction')

		n += 1
		save_batch_results(model_output, n, args)


def cluster_features(model, args):
	similarity_data = compute_similarity(model, args)
	torch.save(similarity_data, f'{args.sys_path}/similarity_matrix.pckl')

	write_rttms(similarity_data, args)


def write_rttms(similarity_data, args):
	threshold = 0.14
	cluster = AHC_Cluster(threshold)
	rttm_list = []

	for video_id in similarity_data['similarities']:
		similarity = similarity_data['similarities'][video_id]
		starts = similarity_data['starts'][video_id]
		ends = similarity_data['ends'][video_id]
		labels = cluster.fit_predict(similarity)

		starts, ends, labels = merge_frames(starts, ends, labels)

		lines = []
		for label, start, end in zip(labels, starts, ends):
			if start < 0: start = 0
			if end - start < 0.01: continue

			lines.append(f'SPEAKER {video_id} 1 {start:010.6f} {(end-start):010.6f} <NA> <NA> spk{label:02d} <NA> <NA>\n')

		pred_path = f'{args.sys_path}/predictions/{video_id}.rttm'
		rttm_list.append(pred_path + '\n')

		with open(pred_path, 'w') as file:
			file.writelines(lines)

	with open(f'{args.sys_path}/{args.data_type}.out', 'w') as file:
		file.writelines(rttm_list)


def merge_frames(starts, ends, labels):
    # Sort the segments by their start times
    sorted_indices = np.argsort(starts)
    starts, ends, labels = starts[sorted_indices], ends[sorted_indices], labels[sorted_indices]

    # Identify adjacent or overlapping segments with the same label
    adjacent_or_overlapping = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    different_labels_or_separated = np.logical_or(~adjacent_or_overlapping, labels[1:] != labels[:-1])
    split_indices = np.nonzero(different_labels_or_separated)[0]

    # Merge the segments based on the identified split points
    merged_starts = starts[np.r_[0, split_indices + 1]]
    merged_ends = ends[np.r_[split_indices, -1]]
    merged_labels = labels[np.r_[0, split_indices + 1]]

    # Adjust for overlapping segments by averaging the overlap points
    overlapping_indices = np.nonzero(merged_starts[1:] < merged_ends[:-1])[0]
    merged_ends[overlapping_indices] = merged_starts[overlapping_indices + 1] = (
        merged_ends[overlapping_indices] + merged_starts[overlapping_indices + 1]
    ) / 2.0

    return merged_starts, merged_ends, merged_labels


def compute_similarity(model, args):
	similarities = {}
	starts = {}
	ends = {}

	batch_size = 255
	model.eval()

	for video_id in tqdm(args.video_ids, desc='Clustering features'):
		utterances = features_by_video(video_id, args)
		utterance_count = len(utterances)
		batch = []
		similarity = torch.diag_embed(torch.ones([utterance_count]))

		total_elements = int((utterance_count * (utterance_count - 1)) / 2)
		progress_bar = tqdm(range(total_elements), leave=False, desc=f'Processing {utterance_count} utterances')

		for i in range(utterance_count):
			for j in range(i + 1, utterance_count):
				batch.append({
					'video_features': torch.cat((utterances[i]['video_features'], utterances[j]['video_features'])),
					'audio_features': torch.cat((utterances[i]['audio_features'], utterances[j]['audio_features'])),
					'index_a': i,
					'index_b': j,
					'visible_a': int(utterances[i]['visible']),
					'visible_b': int(utterances[j]['visible'])
				})

				if len(batch) == batch_size:
					similarity = process_one_batch(batch, model, similarity)
					progress_bar.update(len(batch))
					batch = []

			if similarity[i, -1] != 0:
				del utterances[i]['video_features']
				del utterances[i]['audio_features']

		if len(batch) > 0:
			similarity = process_one_batch(batch, model, similarity)
			progress_bar.update(len(batch))

		similarities[video_id] = similarity
		starts[video_id] = np.array([utterance['start'] for utterance in utterances])
		ends[video_id] 	= np.array([utterance['end'] for utterance in utterances])

	return {'similarities': similarities, 'starts': starts, 'ends': ends}


def features_by_video(video_id, args):
	features = []

	output = load_features(video_id, args)

	for feat_video, feat_audio, video, start, end, visible, tackid in zip(
		output['feat_video'], output['feat_audio'], output['video'], output['start'], output['end'], output['visible'], output['trackid']
	):
		features.append({
			'video_features':	feat_video,
			'audio_features':	feat_audio,
			'start':					start,
			'end':						end,
			'visible':				visible,
			'track_id':				tackid
		})

	return features


def process_one_batch(batch, model, similarity):
	video = torch.cat([data_pair['video_features'].unsqueeze(0) for data_pair in batch], dim=0)
	audio = torch.cat([data_pair['audio_features'].unsqueeze(0) for data_pair in batch], dim=0)

	task 	= torch.tensor([
		torch.tensor(data_pair['visible_a'] + data_pair['visible_b'], dtype=torch.int64) for data_pair in batch
	], device=video.device)

	task1 = torch.tensor([
		torch.tensor(2 * data_pair['visible_a'] + data_pair['visible_b'], dtype=torch.int64) for data_pair in batch
	], device=video.device)

	task2 = torch.tensor([
		torch.tensor(2 * data_pair['visible_b'] + data_pair['visible_a'], dtype=torch.int64) for data_pair in batch
	], device=video.device)

	prepared_batch = {
		'video': video,
		'audio': audio,
		'task': task,
		'task_full': [task1, task2]
	}

	scores = model(prepared_batch, exec='relation')['scores'].cpu()

	for data_pair, score in zip(batch, scores):
		similarity[data_pair['index_a'], data_pair['index_b']] = similarity[data_pair['index_b'], data_pair['index_a']] = score.cpu()

	return similarity


def save_batch_results(cumulating, n, args):
	with open(f'{args.sys_path}/features/batch_{n}.pckl', 'wb') as file:
		pickle.dump(cumulating, file)


def load_features(video_id, args):
	dicts = []
	file_count = len(glob(f'{args.sys_path}/features/*'))

	for i in range(1, file_count + 1):
		with open(f'{args.sys_path}/features/batch_{i}.pckl', 'rb') as file:
			try:
				dicts.append(pickle.load(file))
			except EOFError:
				print(f'File {i} is empty, skipping...')

	for dic in dicts:
		to_remove = []
		for index, data_id in enumerate(dic['video']):
			if data_id != video_id:
				to_remove.append(index)

		to_remove = reversed(to_remove)
		for index in to_remove:
			dic['feat_audio'] = torch.cat((dic['feat_audio'][:index], dic['feat_audio'][index+1:]))
			dic['feat_video'] = torch.cat((dic['feat_video'][:index], dic['feat_video'][index+1:]))
			dic['video'].pop(index)
			dic['start'].pop(index)
			dic['end'].pop(index)
			dic['trackid'].pop(index)
			dic['visible'].pop(index)

	return merge_features(dicts)


def merge_features(dicts):
	features = dicts[0]

	for batch in dicts[1:]:
		for key, value in batch.items():
			if isinstance(value, list):
				features[key].extend(value)
			elif isinstance(value, torch.Tensor):
				features[key] = torch.cat((features[key], value))
			elif isinstance(value, dict):
				features[key] = merge_features([features[key], value])

	return features


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--data_type',		type=str,	help='Type of data being processed, test, val or train')
	parser.add_argument('--video_ids',		type=str,	help='Video ids separated by commas')
	parser.add_argument('--videos_path',	type=str,	help='Path to the videos to work with')
	parser.add_argument('--waves_path',		type=str,	help='Path to the waves, already denoised')
	parser.add_argument('--labs_path',		type=str,	help='Path to the lab files with voice activity detection info')
	parser.add_argument('--frames_path',	type=str,	help='Path to the face frames already cropped and aligned')
	parser.add_argument('--tracks_path',	type=str,	help='Path to the csv files containing the active speaker detection info')
	parser.add_argument('--weights_path', type=str,	help='Path to the weights to be used for training', default=None)
	parser.add_argument('--sys_path',			type=str,	help='Path to the folder where to save all the system outputs')

	args = argparse_helper(parser, **kwargs)

	args.video_ids = args.video_ids.split(',')

	if args.weights_path is not None:
		CONFIG['checkpoint'] = args.weights_path

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	predict(args)


if __name__ == '__main__':
	args = initialize_arguments()
	predict(args)
