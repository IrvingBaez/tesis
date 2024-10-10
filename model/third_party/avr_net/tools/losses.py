import torch.nn as nn
import torch.distributed as dist


class MSELoss(nn.Module):
	"""Mean Squared Error loss adapted for DDP"""
	def __init__(self):
		super().__init__()
		self.loss_weight = 1.0
		self.loss_fn = nn.MSELoss(reduction='sum')
		self.eval_applicable = False

	def forward(self, scores, targets):
		local_loss = self.loss_fn(scores, targets)
		batch_size = scores.shape[0]

		# FOR MULTIPLE GPUS:
		# # Normalize the loss by the batch size on this GPU
		# local_loss = local_loss / batch_size * self.loss_weight

		# # All-reduce to sum up the losses across all GPUs
		# dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)

		# # Divide by the number of GPUs to average the loss
		# world_size = dist.get_world_size()  # Get the number of GPUs
		# avg_loss = local_loss / world_size

		avg_loss = local_loss / batch_size * self.loss_weight

		return avg_loss.view(1)
