import torch, argparse, pickle, os
import numpy as np

from model.third_party.avr_net.tools.custom_collator import CustomCollator
from model.third_party.avr_net.tools.ahc_cluster import AHC_Cluster
from model.third_party.avr_net.tools.dataset import CustomDataset
from model.util import argparse_helper
from collections import defaultdict
from avr_net import AVRNET
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
	# rmtree(args.save_path)
	# os.makedirs(args.save_path)
	# os.makedirs(f'{args.save_path}/features')
	# os.makedirs(f'{args.save_path}/predictions')

	model = load_model()

	# extract_features(model, args)
	cluster_features(model, args)


def load_model():
	# INSTANTIATE MODEL
	model = AVRNET(CONFIG)
	model.build()

	# LOAD MODEL TO GPU
	if torch.cuda.is_available():
		device = torch.device("cuda")
		torch.cuda.set_device(0)
	else:
		device = torch.device("cpu")

	model.to(device)

	return model


def extract_features(model, args):
	# LOAD DATA
	dataset = CustomDataset(args.data_path, args.detector)
	dataset.load_dataset()

	# Set shuffle=true for training
	dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, collate_fn=CustomCollator())

	# INSTANTIATE OPTIMIZER
	parameters = list(model.parameters())
	optimizer = torch.optim.Adam(parameters, lr=0.0005, weight_decay=0.0001)
	scaler = torch.cuda.amp.GradScaler(enabled=False)

	with torch.no_grad():
		dataloader = tqdm(dataloader)
		model_outputs = {}

		n = 0
		for batch in dataloader:
			batch['frames'] = batch['frames'].to(model.device)
			batch['audio'] = batch['audio'].to(model.device)

			model_output = model(batch, exec='extraction')

			n += 1
			save_batch_results(model_outputs, model_output, n)


def cluster_features(model, args):
	# METRICS: [{'type': 'der', 'datasets': ['avaavd'], 'params': {'threshold': 0.14, 'relation': True, 'save_dir': 'save/token/avaavd', 'ground_truth': './dataset//rttms'}}]
  # METRIC REQUIRED PARAMS: {'video', 'feat_audio', 'start', 'dataset_type', 'feat_video', 'end', 'trackid', 'dataset_name', 'visible'}

	# Callbacks: lr_schedueler, checkpoint, logistics

	features = load_features(args)
	similarity_data = compute_similarity(model, features, args)
	torch.save(similarity_data, f'{args.save_path}/similarity_matrix.pckl')

	write_rttms(similarity_data, args)


def write_rttms(similarity_data, args):
	threshold = 0.14
	cluster = AHC_Cluster(threshold)

	for video_id in similarity_data['similarities']:
		similarity = similarity_data['similarities'][video_id]
		labels = cluster.fit_predict(similarity)
		starts = similarity_data['starts'][video_id]
		ends = similarity_data['ends'][video_id]

		lines = []
		for label, start, end in zip(labels, starts, ends):
			if end - start < 0.01: continue

			lines.append(f'SPEAKER {video_id} 1 {start:.6f} {(end-start):.6f} <NA> <NA> {label} <NA> <NA>\n')

		pred_path = f'{args.save_path}/predictions/{video_id}.rttm'

		with open(pred_path, 'w') as file:
			file.writelines(lines)


def compute_similarity(model, output, args):
	similarities = {}
	starts = {}
	ends = {}

	feat_by_video = defaultdict(list)

	for feat_video, feat_audio, video_id, start, end, visible, tackid in zip(
		output['feat_video'], output['feat_audio'], output['video'], output['start'], output['end'], output['visible'], output['trackid']
	):
		feat_by_video[video_id[0]].append({
			'video_features':	feat_video,
			'audio_features':	feat_audio,
			'start':					start,
			'end':						end,
			'visible':				visible,
			'track_id':				tackid
		})

	batch_size = 64
	model.eval()
	del output

	for video_id in tqdm(list(feat_by_video.keys())):
		utterances = feat_by_video[video_id]
		utterance_count = len(utterances)
		batch = []
		similarity = torch.diag_embed(torch.ones([utterance_count]))

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
					process_one_batch(batch, model, similarity)
					batch = []

		if len(batch) > 0:
			process_one_batch(batch, model, similarity)

		del feat_by_video[video_id]

		similarities[video_id] = similarity
		starts[video_id] = np.array([utterance['start'] for utterance in utterances])
		ends[video_id] 	= np.array([utterance['end'] for utterance in utterances])

	return {'similarities': similarities, 'starts': starts, 'ends': ends}


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


def save_batch_results(cumulating, n, args):
	with open(f'{args.save_path}/features/batch_{n}.pckl', 'wb') as file:
		pickle.dump(cumulating, file)


def load_features(args):
	dicts = []
	file_count = len(glob(f'{args.save_path}/features/*'))

	for i in range(1, file_count + 1):
		with open(f'{args.save_path}/features/batch_{i}.pckl', 'rb') as file:
			try:
				dicts.append(pickle.load(file))
			except EOFError:
				print(f'File {i} is empty, skipping...')


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

	parser.add_argument('--data_path',	type=str, default="dataset/val",   help='Path to the main folder with the data')
	parser.add_argument('--detector',		type=str, default="ground_truth",   help='ASD detector being evaluated')

	args = argparse_helper(parser, **kwargs)

	args.save_path = f'{args.data_path}/avd/avr_net'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	predict(args)


if __name__ == '__main__':
	args = initialize_arguments()
	predict(args)
