import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin


	def forward(self, x_1, x_2, y):
		B, C, F, H, W = x_1.shape

		x_1 = x_1.reshape(B, C, F, H*W)
		x_2 = x_2.reshape(B, C, F, H*W)
		x_1 = normalize(x_1)
		x_2 = normalize(x_2)

		# TODO: Is this the correct way to calculate distance?
		distance = torch.linalg.matrix_norm((x_1-x_2))
		distance /= torch.max(distance)

		margins = self.margin * torch.ones(distance.shape, device=distance.device)
		ones = torch.ones(y.shape, device=y.device)

		# y=1 for similar pairs. y=0 for disimilar pairs
		# Positive label. All distances add up. Minimizes loss by minimizing distance.
		pos = y*(1/2)*(distance)**2

		# Negative label. Only distances less than the margin add up. Minimizes loss by maximizing distance.
		neg = (ones - y)*(1/2)*(torch.clamp(margins - distance, min=0.0))**2

		loss = torch.mean(neg + pos)

		return loss


# Set mean=0, s=1
def normalize(tensor):
	mean = tensor.mean(dim=[0,2,3], keepdim=True)
	std = tensor.std(dim=[0,2,3], keepdim=True)

	return (tensor - mean) / std
