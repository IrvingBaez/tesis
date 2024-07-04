import torch
from .timer import Timer


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

		# TODO: set False after finishing debugging.
		torch.autograd.set_detect_anomaly(True)


	def train(self) -> None:
		self.run_training_epoch()

	def run_training_epoch(self) -> None:
		while self.current_updates < self.max_updates:
			self.current_epoch += 1

			for index, batch in enumerate(self.dataloader):
				batch['frames'] = batch['frames'].transpose(0, 1)
				batch['frames'] = batch['frames'].to(self.model.device)

				batch['audio'] = batch['audio'].transpose(0, 1)
				batch['audio'] = batch['audio'].to(self.model.device)

				batch['targets'] = batch['targets'].to(self.model.device)

				model_output = self.model(batch, exec='train')
				print(model_output)
