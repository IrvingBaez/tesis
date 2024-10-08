import torch
import torch.nn as nn

class PickFirst(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x = x[0]
		x = x.reshape(1, *x.shape)

		return x


class Identity(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x


class CrossIdentity(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, video, audio):
		return video, audio


class AudioVisualAttention(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.bottle_neck = nn.Linear(512, 256)
		self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

	def forward(self, audio, video):
		# video: [2, 40, 512, 7, 7]
		# audio: [2, 40, 256, 7, 7]

		N, T, C_v, H, W = video.shape
		_, _, C_a, _, _ = audio.shape

		# Project dim 2 to size 256.
		video = video.view(N, T, C_v, H * W)  # [N, T, 512, 49]
		video = video.permute(0, 1, 3, 2)  # [N, T, 49, 512]
		video = self.bottle_neck(video)
		video = video.permute(0, 1, 3, 2).view(N, T, 256, H, W)  # [N, T, 256, 7, 7]

		# Flatten spacial dimensions
		video = video.view(N, T, 256, H * W)  # [N, T, 256, 49]
		audio = audio.view(N, T, 256, H * W)  # [N, T, 256, 49]

		# Reshape to (L, N, E): Lenght, Number, Embeddings
		video = video.permute(3, 0, 2, 1).reshape(H * W, N * T, 256)  # [49, N*T, 256]
		audio = audio.permute(3, 0, 2, 1).reshape(H * W, N * T, 256)  # [49, N*T, 256]

		output, _ = self.cross_attention(query=audio, key=video, value=video)

		# Back to original shape (N, T, 256, H, W)
		output = output.view(H * W, N, T, 256).permute(1, 2, 3, 0).view(N, T, 256, H, W)

		return output
