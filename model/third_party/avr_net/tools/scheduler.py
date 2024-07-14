from bisect import bisect, bisect_right
from torch.optim.lr_scheduler import LambdaLR


class MultiStepScheduler(LambdaLR):
	def __init__(self, optimizer):
		self.global_lr_ratio = 0.1
		self.global_lr_steps = []

		self.use_warmup = False
		self.lr_steps = [40000, 80000]
		self.lr_ratio = 0.5
		self.warmup_iterations = 0
		self.warmup_factor = 0.2

		super().__init__(optimizer, self.lr_lambda)


	def lr_lambda(self, step):
		idx = bisect(self.global_lr_steps, step)

		return pow(self.global_lr_ratio, idx)


	def get_lr(self):
		return [0.0005 * self.lr_ratio ** bisect_right(self.lr_steps, self.last_epoch)]
