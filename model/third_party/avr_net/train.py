import argparse
import os
import torch
from shutil import rmtree
from datetime import datetime
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model.util import argparse_helper, save_data, get_path
from .models.relation_layer import RelationLayer
from .tools.custom_collator import CustomCollator
from .tools.dataset import CustomDataset
from .tools.clustering_dataset import ClusteringDataset
from .tools.feature_extractor import FeatureExtractor
from .tools.attention import AudioVisualAttention
from .tools.losses import MSELoss

def train(args):
	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(f'{args.sys_path}')

	with torch.no_grad():
		train_features = extract_features(args, mode='train')
		save_data(train_features, args.train_features_path)
		val_features = extract_features(args, mode='vali')
		save_data(val_features, args.val_features_path)

	attention_model, relation_model = load_models(args)
	attention_optimizer, relation_optimizer = load_optimizers(attention_model, relation_model, args)
	# TODO: Add schedueler
	# scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)
	# scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)

	# # Al final de cada época
	# scheduler1.step()
	# scheduler2.step()

	criterion = MSELoss()

	start_epoch, train_losses, val_losses = load_checkpoint()
	total_epochs = start_epoch + args.epochs

	train_dataloader = load_data(args.train_features_path)
	val_dataloader = load_data(args.val_features_path)

	for epoch in tqdm(range(start_epoch, total_epochs), initial=start_epoch, total=total_epochs, desc='Training'):
		# ======================== Training Phase ========================
		attention_model.train()
		relation_model.train()
		train_loss = 0.0

		for batch in tqdm(train_dataloader, desc='Training', leave=False):
			audio = attention_model(batch['video'], batch['audio'])
			scores = relation_model(batch['video'], audio, batch['task_full'])

			loss += criterion(scores, batch['targets'])
			train_loss += loss.item()

			attention_optimizer.zero_grad()
			relation_optimizer.zero_grad()

			train_loss.backward()

			attention_optimizer.step()
			relation_optimizer.step()

		train_losses.append(train_loss)

		# ======================== Validation Phase ========================
		attention_model.eval()
		relation_model.eval()
		val_loss = 0.0

		for batch in tqdm(val_dataloader, desc='Validating', leave=False):
			video, audio = attention_model(batch['video'], batch['audio'])
			scores = relation_model(video, audio, batch['task_full'])

			loss += criterion(scores, batch['targets'])
			val_loss += loss.item()

		val_losses.append(val_loss)

		# ======================== Saving data ========================
		print(f'Epoch: {epoch}\tTrain Loss: {train_loss}\tVal Loss:{val_loss}')

		timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
		checkpoint_path = f'{args.checkpoint_dir}/attention_{timestamp}_epoch_{epoch:05d}.ckpt'

		save_data({
			'attention_model_state_dict': 		attention_model.state_dict(),
			'relation_model_state_dict': 			relation_model.state_dict(),
			'attention_optimizer_state_dict': attention_optimizer.state_dict(),
			'relation_optimizer_state_dict': 	relation_optimizer.state_dict(),
			'train_losses': train_losses,
			'val_losses': val_losses,
			'epoch': epoch
		}, checkpoint_path)

	graph_losses(train_losses, val_losses,f'{args.checkpoint_dir}/attention_{timestamp}_epoch_{epoch:05d}.png')
	rmtree(f'{args.sys_path}/features')


def graph_losses(train_losses, val_losses, filename):
	plt.figure(figsize=(10, 5))
	plt.plot(train_losses, label='train losses')
	plt.plot(val_losses, label='val losses')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Training and Validation Losses')
	plt.legend()
	plt.grid(True)
	plt.savefig(filename)
	plt.close()


def extract_features(args, mode):
	feature_extractor = FeatureExtractor()
	feature_extractor.to(args.device)

	if mode == 'train':
		config = args.train_dataset_config
	else:
		config = args.val_dataset_config

	dataset = CustomDataset(config, training=True)
	dataloader = DataLoader(dataset, batch_size=255, shuffle=False, num_workers=1, pin_memory=True, drop_last=False, collate_fn=CustomCollator())
	dataloader = tqdm(dataloader, desc='Extracting features')

	feature_list = []
	for batch in dataloader:
		feature_list.append(feature_extractor(batch))

	features = merge_features(feature_list)

	return features


def load_data(features_path):
	dataset = ClusteringDataset(features_path)
	dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, collate_fn=CustomCollator())

	return dataloader


def load_optimizers(attention_model, relation_model, args):
	attention_optimizer = Adam(attention_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	relation_optimizer = Adam(relation_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	return attention_optimizer, relation_optimizer


def load_models(args):
	attention_model = AudioVisualAttention()
	attention_model.to(args.device)
	attention_model.train()

	relation_model = RelationLayer(args.checkpoint)
	relation_model.to(args.device)
	relation_model.train()

	return attention_model, relation_model


def load_checkpoint(checkpoint_path=None, model=None):
	start_epoch, train_losses, val_losses = 0, [], []
	if not checkpoint_path:
		print(f'No checkpoint provided, using default initialization')
		return start_epoch, train_losses, val_losses

	if not os.path.isfile(checkpoint_path):
		print(f'Checkpoint not found at: {checkpoint_path}, using default initialization')
		return start_epoch, losses

	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])

	# if 'optimizer_state_dict' in checkpoint:
	# 	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	# if 'schedueler_state_dict' in checkpoint:
	# 	schedueler.load_state_dict(checkpoint['schedueler_state_dict'])

	if 'epoch' in checkpoint:
		start_epoch = checkpoint['epoch'] + 1

	if 'losses' in checkpoint:
		losses = checkpoint['losses']

	if not isinstance(losses, list): losses = [losses]

	print(f'Loading checkpoint from {checkpoint_path}, resuming training from epoch {start_epoch}')

	return start_epoch, losses


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
	parser.add_argument('--gpu_batch_size',	type=int,	help='Training batch size per GPU', default=4)
	parser.add_argument('--learning_rate',	type=int,	help='Training learning rate', default=0.0005)
	parser.add_argument('--weight_decay',		type=int,	help='Training weight decay', default=0.0001)
	parser.add_argument('--epochs', 				type=int, help='Epochs to add to the training of the checkpoint', default=100)

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
	args.checkpoint_dir = f'model/third_party/avr_net/checkpoints/{args.relation_layer}'
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
