import torch, argparse, pickle, os
import numpy as np
import torch.nn as nn

from .tools.predict_collator import PredictCollator
from .tools.train_collator import TrainCollator
from .tools.train_dataset import TrainDataset
from .tools.train_sampler import TrainSampler
from .tools.trainer import Trainer
from .avr_net import AVRNET
from model.util import argparse_helper
from shutil import rmtree

from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp


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


def train(rank, world_size, args):
	setup(rank, world_size)

	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(args.sys_path)
	os.makedirs(f'{args.sys_path}/features')

	model, device = load_model(rank)
	model.train()

	dataset = TrainDataset(args)
	dataset.load_dataset()

	# sampler = TrainSampler(dataset)

	# gpu_count = torch.cuda.device_count()
	# dataloader = DataLoader(
	# 	dataset,
	# 	batch_size=1*gpu_count,
	# 	num_workers=gpu_count,
	# 	sampler=sampler,
	# 	collate_fn=TrainCollator(),
	# 	pin_memory=True,
	# 	drop_last=False,
	# 	persistent_workers=True
	# )

	dataloader = create_dataloader(rank, world_size, dataset, batch_size=4)

	trainer = Trainer(model, device, dataloader, rank)
	trainer.train(CONFIG['checkpoint'])

	rmtree(f'{args.sys_path}/features')
	cleanup()


def setup(rank, world_size):
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'

	dist.init_process_group("nccl", rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)


def cleanup():
	dist.destroy_process_group()


def create_dataloader(rank, world_size, dataset, batch_size):
	# FIXME: ES EL SAMPLER, DEBERÏA USAR TainSampler >:v!!!
	# sampler = TrainSampler(dataset)
	# sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
	sampler = TrainSampler(dataset, num_replicas=world_size, rank=rank)
	dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, collate_fn=TrainCollator())

	return dataloader


def load_model(rank):
	# INSTANTIATE MODEL
	model = AVRNET(CONFIG)
	model.build()

	# LOAD MODEL TO GPU
	if torch.cuda.is_available():
		device = torch.device(f'cuda:{rank}')
		# torch.cuda.set_device(0)
	else:
		device = torch.device("cpu")

	model.to(device)
	model= DDP(model, device_ids=[rank])

	return model, device


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--video_ids',		type=str,	help='Video ids separated by commas')
	parser.add_argument('--videos_path',	type=str,	help='Path to the videos to work with')
	parser.add_argument('--waves_path',		type=str,	help='Path to the waves, already denoised')
	parser.add_argument('--labs_path',		type=str,	help='Path to the lab files with voice activity detection info')
	parser.add_argument('--frames_path',	type=str,	help='Path to the face frames already cropped and aligned')
	parser.add_argument('--tracks_path',	type=str,	help='Path to the csv files containing the active speaker detection info')
	parser.add_argument('--rttms_path', 	type=str,	help='Path to the rttm files containing detection ground truth')
	parser.add_argument('--weights_path', type=str,	help='Path to the weights to be used for training', default=None)
	parser.add_argument('--sys_path',			type=str,	help='Path to the folder where to save all the system outputs')

	args = argparse_helper(parser, **kwargs)

	args.data_type = 'train'
	args.video_ids = args.video_ids.split(',')

	if args.weights_path is not None:
		CONFIG['checkpoint'] = args.weights_path

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	world_size = torch.cuda.device_count()

	mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
	#	train(args)


if __name__ == '__main__':
	args = initialize_arguments()
	world_size = torch.cuda.device_count()

	mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
	# train(args)