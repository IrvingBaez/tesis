import torch, argparse, pickle, os
import numpy as np


from .tools.predict_collator import PredictCollator
from .tools.train_collator import TrainCollator
from .tools.ahc_cluster import AHC_Cluster
from .tools.train_dataset import TrainDataset
from .tools.train_sampler import TrainSampler
from .tools.trainer import Trainer
from .avr_net import AVRNET
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
		'num_shot': 2,
		'num_filters': [256, 64]
	},
	'checkpoint': 'model/third_party/avr_net/weights/best_0.14_20.66.ckpt'
}


def train(args):
	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(args.sys_path)
	os.makedirs(f'{args.sys_path}/features')

	model = load_model()
	model.train()

	dataset = TrainDataset(args)
	dataset.load_dataset()

	sampler = TrainSampler(dataset)

	dataloader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True, drop_last=False, collate_fn=PredictCollator(), sampler=sampler)
	trainer = Trainer(model, dataloader)

	trainer.train()

	rmtree(f'{args.sys_path}/features')


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


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--video_ids',		type=str,	help='Video ids separated by commas')
	parser.add_argument('--videos_path',	type=str,	help='Path to the videos to work with')
	parser.add_argument('--waves_path',		type=str,	help='Path to the waves, already denoised')
	parser.add_argument('--labs_path',		type=str,	help='Path to the lab files with voice activity detection info')
	parser.add_argument('--frames_path',	type=str,	help='Path to the face frames already cropped and aligned')
	parser.add_argument('--tracks_path',	type=str,	help='Path to the csv files containing the active speaker detection info')
	parser.add_argument('--rttms_path',		type=str,	help='Path to the rttm files containing detection ground truth')
	parser.add_argument('--sys_path',			type=str,	help='Path to the folder where to save all the system outputs')

	args = argparse_helper(parser, **kwargs)

	args.data_type = 'train'
	args.video_ids = args.video_ids.split(',')

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	train(args)