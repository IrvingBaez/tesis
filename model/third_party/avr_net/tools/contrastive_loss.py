import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
	def __init__(self, pos_margin=0.0, neg_margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.pos_margin = pos_margin
		self.neg_margin = neg_margin


	def forward(self, scores, y):
		pos_margins = self.pos_margin * torch.ones(scores.shape, device=scores.device)
		neg_margins = self.neg_margin * torch.ones(scores.shape, device=scores.device)

		# 1 = similar or 0 distance. 0 = different, or 1 distance.
		distance = 1 - scores
		target = 1 - y

		# y=1 for similar pairs. y=0 for disimilar pairs
		# Positive label, target is 0. All distances add up. Minimizes loss by minimizing distance.
		pos = (1-target)*(1/2)*(torch.clamp(distance - pos_margins, min=0.0))**2

		# Negative label, target is 1. Only distances less than the margin add up. Minimizes loss by maximizing distance.
		neg = (target)*(1/2)*(torch.clamp(neg_margins - distance, min=0.0))**2

		loss = torch.mean(neg + pos)

		return loss