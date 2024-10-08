import numpy as np
import torch


class FacePad:
	def __init__(self, config:dict=None):
		self.length = config['length']

	def __call__(self, item):
		if item['frames'].shape[0] == 0:
			item['frames'] = np.zeros((self.length, 224, 224, 3), dtype=np.uint8)

		if item['frames'].shape[0] < self.length:
			pad_width = ((0, self.length - item['frames'].shape[0]), (0, 0), (0, 0), (0, 0))
			item['frames'] = np.pad(item['frames'], pad_width=pad_width, mode='edge')

		return item


class FaceToTensor:
	def __init__(self, config:dict=None):
		return

	def __call__(self, item):
		frames = [torch.tensor(frame).unsqueeze(0) for frame in item['frames']]
		frames = torch.cat(frames, dim=0)
		item['frames'] = frames.permute(3, 0, 1, 2).to(torch.float32) / 255

		return item


class FaceResize:
	def __init__(self, config:dict=None):
		self.dest_size = list(config['dest_size'])

	def __call__(self, item):
		scale = None

		if isinstance(self.dest_size, int):
			scale = float(self.dest_size) / min(item['frames'].shape[-2:])
			self.dest_size = None

		item['frames'] = torch.nn.functional.interpolate(
			item['frames'], size=self.dest_size, scale_factor=scale, mode='bilinear', align_corners=False
		)

		return item


class FaceNormalize:
	def __init__(self, config:dict=None):
		self.mean = config['mean']
		self.std = config['std']

	def __call__(self, item):
		shape = (-1,) + (1,) * (item['frames'].dim() - 1)
		mean = torch.as_tensor(self.mean).reshape(shape)
		std = torch.as_tensor(self.std).reshape(shape)

		item['frames'] = (item['frames'] - mean) / std

		return item


class AudioNormalize:
	def __init__(self, config:dict=None):
		self.desired_rms = config['desired_rms']
		self.eps = config['eps']

	def __call__(self, item):
		audio = item['audio']

		if isinstance(audio, torch.Tensor):
				audio = audio.numpy()
		elif not isinstance(audio, np.ndarray):
				raise TypeError('Invalid audio data type')

		rms = np.maximum(self.eps, np.sqrt(np.mean(audio**2)))
		item['audio'] = audio * (self.desired_rms / rms)

		return item


class AudioToTensor:
	def __init__(self, config:dict=None):
		return

	def __call__(self, item):
		if isinstance(item['audio'], np.ndarray):
			item['audio'] = torch.FloatTensor(item['audio'])

		return item
