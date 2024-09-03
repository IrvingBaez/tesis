import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model.util import argparse_helper

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# Configuración de parámetros
batch_size = 64  # Prueba con un tamaño de batch más pequeño
learning_rate = 0.001
world_size = torch.cuda.device_count()
data_path = 'model/third_party/pytorch_parallel/data'
checkpoint_dir = 'model/third_party/pytorch_parallel/checkpoints'
torch.backends.cudnn.benchmark = False

# Establecer variable de entorno para evitar problemas con NCCL
os.environ['NCCL_P2P_DISABLE'] = '1'

def setup(rank, world_size):
	# Establecer las variables de entorno necesarias
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

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


def download_dataset():
	# Esta función se llama solo en el proceso principal para descargar el dataset
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)


def save_checkpoint(epoch, model, optimizer, rank):
	# Solo el proceso con rank 0 guarda el checkpoint
	if rank == 0:
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:05d}.pth')

		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
		}, checkpoint_path)
		print(f'Checkpoint guardado en {checkpoint_path}')


def load_checkpoint(checkpoint_path, model, optimizer):
	if not checkpoint_path: return 0

	if os.path.isfile(checkpoint_path):
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch'] + 1

		print(f'Checkpoint cargado desde {checkpoint_path}, reanudando desde la epoch {start_epoch}')

		return start_epoch
	else:
		print(f'No se encontró el checkpoint en la ruta {checkpoint_path}')

		return 0


def train(rank, world_size, args):
	setup(rank, world_size)

	# Transformaciones para los datos de entrada
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# Cargar el dataset CIFAR-10
	train_dataset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
	train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)

	# Definir el modelo (usando ResNet18 como ejemplo)
	model = models.resnet18().cuda(rank)
	model = DDP(model, device_ids=[rank])

	# Definir el criterio de pérdida y el optimizador
	criterion = nn.CrossEntropyLoss().cuda(rank)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	# Cargar checkpoint si se proporciona
	start_epoch = load_checkpoint(args.checkpoint, model, optimizer)
	total_epochs = start_epoch + args.epochs

	# Entrenamiento
	for epoch in range(start_epoch, total_epochs):
		train_sampler.set_epoch(epoch)
		model.train()
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.cuda(rank, non_blocking=True), targets.cuda(rank, non_blocking=True)

			optimizer.zero_grad()
			outputs = model(inputs)

			# Sincronizar GPUs antes de cualquier operación crítica
			torch.cuda.synchronize(rank)

			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			# Sincronizar nuevamente después de la actualización
			torch.cuda.synchronize(rank)

			if batch_idx % 10 == 0 and rank == 0:
				print(f'Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item()}')

		# Guardar un checkpoint al final de cada epoch
		save_checkpoint(epoch, model, optimizer, rank)

	cleanup()


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Example of pytorch training on multiple GPUs")

	parser.add_argument('--checkpoint', type=str,	help='Path of checkpoint to continue training')
	parser.add_argument('--epochs', type=int, default=5,	help='Epochs to add to the training of the checkpoint')

	args = argparse_helper(parser, **kwargs)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)

	# Descargar el dataset en el proceso principal (rank 0)
	if not os.path.exists(f'{data_path}/cifar-10-batches-py'):
		download_dataset()

	# Usar mp.spawn para lanzar múltiples procesos
	mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
	args = initialize_arguments()
	main(**vars(args))
