import torch.nn as nn


class MSELoss(nn.Module):
	"""Mean Squared Error loss"""
	def __init__(self):
		super().__init__()
		self.loss_weight = 1.0
		self.loss_fn = nn.MSELoss(reduction='sum')
		self.eval_applicable = False


	def forward(self, scores, targets):
		batch_size = scores.shape[0]

		return (self.loss_fn(scores, targets) / batch_size * self.loss_weight).view(1)