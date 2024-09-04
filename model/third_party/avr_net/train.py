import argparse
import os
import time
import torch

from .avr_net import AVRNET
from .tools.scheduler import MultiStepScheduler
from .tools.train_collator import TrainCollator
from .tools.train_dataset import TrainDataset
from .tools.train_sampler import TrainSampler
# from .tools.trainer import Trainer
from .tools.losses import MSELoss
from model.util import argparse_helper
from shutil import rmtree

from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


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


def setup(rank, world_size):
	torch.backends.cudnn.benchmark = False

	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)

	torch.cuda.set_device(rank)
	torch.cuda.empty_cache()
	time.sleep(rank * 5)


def cleanup():
	dist.destroy_process_group()


def train(rank, world_size, args):
	setup(rank, world_size)

	# Prepare directories
	if os.path.exists(args.sys_path):
		rmtree(args.sys_path)

	os.makedirs(f'{args.sys_path}/features')

	# Load data
	dataset = TrainDataset(args)
	sampler = TrainSampler(dataset, num_replicas=world_size, rank=rank)
	dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.gpu_batch_size, num_workers=args.world_size, pin_memory=True, collate_fn=TrainCollator())

	# Load prepare model and optimizer data
	model, optimizer_params, device = load_model(rank)
	optimizer = Adam(optimizer_params, lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = MultiStepScheduler(optimizer)
	criterion = MSELoss()
	start_epoch, losses = load_checkpoint(args.checkpoint, model, optimizer, scheduler)
	total_epochs = start_epoch + args.epochs
	model.train()

	epochs_pb = tqdm(initial=start_epoch, total=total_epochs, desc='Training')
	for epoch in range(start_epoch, total_epochs):
		sampler.set_epoch(epoch)

		batches_pb = tqdm(total=len(dataloader), desc='Batches', leave=False)
		for batch in dataloader:
			# batch = self._mount_batch(batch, device)
			model_output = model(batch, exec='train')

			# Synchronize GPUs before grad step
			torch.cuda.synchronize(rank)

			batch.update(model_output)
			loss = criterion(batch['scores'], batch['targets'])
			losses.append(loss)
			loss.backward()

			optimizer.step()
			scheduler.step()

			# Synchronize GPUs after grad step
			torch.cuda.synchronize(rank)

			if rank == 0:
				batches_pb.update()

		if rank == 0:
			epochs_pb.update()
			save_checkpoint(epoch, model, optimizer, scheduler, losses)

	rmtree(f'{args.sys_path}/features')
	cleanup()


def load_model(rank):
	device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else "cpu")
	model = AVRNET(CONFIG)
	model.build(device)
	model.to(device)
	model = DDP(model, device_ids=[rank])
	torch.cuda.synchronize()

	optimizer_params = [{"params": list(model.parameters())}]

	return model, optimizer_params, device


def save_checkpoint(epoch, model, optimizer, scheduler, losses):
	save_dir = 'model/third_party/avr_net/checkpoints'
	os.makedirs(save_dir, exist_ok=True)

	timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
	checkpoint_path = f'{save_dir}/training_{timestamp}_epoch_{epoch:05d}.ckpt'

	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'schedueler_state_dict': scheduler.state_dict(),
		'losses': losses
	}, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, schedueler):
	if not checkpoint_path: return 0, []

	if os.path.isfile(checkpoint_path):
		checkpoint = torch.load(checkpoint_path, map_location='cpu')

		# Ensuring full precision
		model_weights = checkpoint['model_state_dict']
		for key, value in model_weights.items():
			model_weights[key] = value.float()

		model.load_state_dict(model_weights)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		schedueler.load_state_dict(checkpoint['schedueler_state_dict'])

		start_epoch = checkpoint['epoch'] + 1
		losses = checkpoint['losses']
		if not isinstance(losses, list): losses = [losses]

		print(f'Loading checkpoint from {checkpoint_path}, resuming training from epoch {start_epoch}')

		return start_epoch, losses
	else:
		print(f'Checkpoint not found at: {checkpoint_path}, using random initialization')

		return 0, []


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
	parser.add_argument('--checkpoint', 		type=str,	help='Path of checkpoint to continue training', default=None)
	parser.add_argument('--epochs', 				type=int, help='Epochs to add to the training of the checkpoint', default=100)

	args = argparse_helper(parser, **kwargs)

	args.data_type = 'train'
	args.video_ids = args.video_ids.split(',')

	if args.checkpoint is not None:
		CONFIG['checkpoint'] = args.checkpoint

	if torch.cuda.is_available():
		args.world_size = torch.cuda.device_count()
	else:
		args.world_size = 1

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
