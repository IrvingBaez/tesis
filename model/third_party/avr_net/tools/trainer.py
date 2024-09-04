import torch, os
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from tqdm.auto import tqdm

from .scheduler import MultiStepScheduler
from .losses import MSELoss


class Trainer:
	def __init__(self, model, device, dataloader, rank) -> None:
		self.model = model
		self.device = device
		self.dataloader = dataloader
		self.rank = rank
		self.max_updates = 60_000
		self.max_epochs = self.max_updates / len(dataloader)

		self.current_updates = 0
		self.current_epoch = 0

		self.loss_function = MSELoss()
		# self.scaler = torch.cuda.amp.GradScaler(enabled=False)
		self.losses = []

		optimizer_params = self._get_optimizer_params()
		self.optimizer = Adam(optimizer_params, lr=0.0005, weight_decay=0.0001)
		self.scheduler = MultiStepScheduler(self.optimizer)

		torch.autograd.set_detect_anomaly(False)


	def train(self, checkpoint_path) -> None:
		checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
		print(f'Process {self.rank} loading weights at {checkpoint_path}')

		model_weights = self._wrap_weights(checkpoint_data)
		self.model.load_state_dict(model_weights)
		self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

		self.current_updates = checkpoint_data['num_updates']
		self.current_epoch = int(checkpoint_data['num_updates'] / len(self.dataloader))
		self.losses = checkpoint_data['losses'] if isinstance(self.losses, list) else [self.losses]

		print(f'Model has {self.current_updates} updates, {max(0, self.max_updates - self.current_updates)} more updates to go...')

		self.run_training_loop()


	def run_training_loop(self) -> None:
		train_pb = tqdm(total=self.max_updates, desc='Training', initial=self.current_updates)

		while self.current_updates < self.max_updates and self.current_epoch < self.max_epochs:
			self.current_epoch += 1
			self.optimizer.zero_grad()

			for batch in self.dataloader:
				batch = self._mount_batch(batch)

				# with torch.cuda.amp.autocast():
				model_output = self.model(batch, exec='train')
				batch.update(model_output)
				batch['loss'] = self.loss_function(batch['scores'], batch['targets'])

				torch.cuda.synchronize()
				self.scaler.scale(batch['loss']).backward()
				torch.cuda.synchronize()

				self._detatch_batch(batch)
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.current_updates += 1
				self.scheduler.step()

				train_pb.update()
				if self.current_updates >= self.max_updates: break

			self._save_checkpoint()


	def _mount_batch(self, batch):
		batch['frames'] = batch['frames'].to(self.device)
		batch['audio'] = batch['audio'].to(self.device)
		batch['targets'] = batch['targets'].to(self.device)

		return batch


	def _detatch_batch(self, batch):
		for value in batch.values():
			if isinstance(value, torch.Tensor):
				value.detach()


	def _get_optimizer_params(self):
		return [{"params": list(self.model.parameters())}]


	def _save_checkpoint(self):
		if self.rank != 0: return

		save_dir = 'model/third_party/avr_net/checkpoints'
		os.makedirs(save_dir, exist_ok=True)

		timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
		checkpoint_path = f'{save_dir}/training_{timestamp}_epoch_{self.current_epoch:05d}.ckpt'

		torch.save({
			'num_updates':					self.current_updates,
			'epoch':								self.current_epoch,
			'model_state_dict':			self.model.module.state_dict(),
			'optimizer_state_dict':	self.optimizer.state_dict(),
			'losses':								self.losses
		}, checkpoint_path)


	def _wrap_weights(self, checkpoint_data):
		state_dict = checkpoint_data['model_state_dict']
		new_state_dict = {}

		for key, value in state_dict.items():
			new_key = f"module.{key}" if not key.startswith("module.") else key
			new_state_dict[new_key] = value

		return new_state_dict
