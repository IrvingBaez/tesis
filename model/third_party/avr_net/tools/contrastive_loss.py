import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin


	def forward(self, distance, y):
		margins = self.margin * torch.ones(distance.shape, device=distance.device)
		ones = torch.ones(y.shape, device=y.device)

		# y=1 for similar pairs. y=0 for disimilar pairs
		# Positive label. All distances add up. Minimizes loss by minimizing distance.
		pos = y*(1/2)*(distance)**2

		# Negative label. Only distances less than the margin add up. Minimizes loss by maximizing distance.
		neg = (ones - y)*(1/2)*(torch.clamp(margins - distance, min=0.0))**2

		loss = torch.mean(neg + pos)

		return loss
