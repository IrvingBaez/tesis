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

# Configuración de parámetros
batch_size = 128
learning_rate = 0.001
epochs = 5
world_size = torch.cuda.device_count()

def setup(rank, world_size):
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=world_size,
		rank=rank
	)
	torch.cuda.set_device(rank)

def cleanup():
	dist.destroy_process_group()

def train(rank, world_size):
	setup(rank, world_size)

	# Transformaciones para los datos de entrada
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# Cargar el dataset CIFAR-10
	train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)

	# Definir el modelo (usando ResNet18 como ejemplo)
	model = models.resnet18().cuda(rank)
	model = DDP(model, device_ids=[rank])

	# Definir el criterio de pérdida y el optimizador
	criterion = nn.CrossEntropyLoss().cuda(rank)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	# Entrenamiento
	for epoch in range(epochs):
		train_sampler.set_epoch(epoch)
		model.train()
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.cuda(rank, non_blocking=True), targets.cuda(rank, non_blocking=True)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			if batch_idx % 10 == 0 and rank == 0:
				print(f'Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item()}')

	cleanup()

def main():
	mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
	main()