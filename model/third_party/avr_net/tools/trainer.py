import torch, os
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from tqdm.auto import tqdm

from .scheduler import MultiStepScheduler
from .timer import Timer
from .losses import MSELoss


class Trainer:
	def __init__(self, model, dataloader) -> None:
		self.train_timer = Timer()
		self.validation_timer = Timer()

		self.model = model
		self.dataloader = dataloader
		self.max_updates = 100_000
		self.max_epochs = self.max_updates / len(dataloader)

		self.current_updates = 0
		self.current_epoch = 0

		self.loss_function = MSELoss()
		self.scaler = torch.cuda.amp.GradScaler(enabled=False)

		optimizer_params = self._get_optimizer_params()
		self.optimizer = Adam(optimizer_params, lr=0.0005, weight_decay=0.0001)
		self.scheduler = MultiStepScheduler(self.optimizer)

		torch.autograd.set_detect_anomaly(False)


	def train(self) -> None:
		self.run_training_epoch()

	def run_training_epoch(self) -> None:
		train_pb = tqdm(total=self.max_updates, desc='Training')

		while self.current_updates < self.max_updates and self.current_epoch < self.max_epochs:
			self.current_epoch += 1
			self.optimizer.zero_grad()

			losses = []
			for batch in self.dataloader:
				batch = self._mount_batch(batch)
				model_output = self.model(batch, exec='train')

				batch.update(model_output)
				batch['loss'] = self.loss_function(batch['scores'], batch['targets'])
				losses.append(batch['loss'])

				self.scaler.scale(batch['loss']).backward()
				self._detatch_batch(batch)

				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.current_updates += 1
				self.scheduler.step()

				train_pb.update()
				if self.current_updates >= self.max_updates: break

			self._save_checkpoint(losses)


	def _mount_batch(self, batch):
		batch['frames'] = batch['frames'].to(self.model.device)
		batch['audio'] = batch['audio'].to(self.model.device)
		batch['targets'] = batch['targets'].to(self.model.device)

		return batch


	def _detatch_batch(self, batch):
		for value in batch.values():
			if isinstance(value, torch.Tensor):
				value.detach()


	def _get_optimizer_params(self):
		parameters = list(self.model.parameters())
		parameters = [{"params": parameters}]

		return parameters


	def _save_checkpoint(self, losses):
		save_dir = 'model/third_party/avr_net/checkpoints'
		os.makedirs(save_dir, exist_ok=True)

		timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
		checkpoint_path = f'{save_dir}/training_{timestamp}.ckpt'

		torch.save({
			'epoch': self.current_epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'losses': losses
		}, checkpoint_path)
