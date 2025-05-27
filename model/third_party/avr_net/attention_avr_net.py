import torch
import torch.nn as nn

from model.third_party.avr_net.models.relation_layer import RelationLayer
import model.third_party.avr_net.tools.attention as attention


class Attention_AVRNet(nn.Module):
	def __init__(self, self_attention, cross_attention, dropout=0.1):
		super().__init__()

		# Set cross attention
		if self_attention == 'class_token':
			self.self_attention 	= attention.SelfAttentionClassToken(dropout=dropout)
		else:
			self.self_attention 	= attention.PickFirst()

		# Set cross attention
		if cross_attention == 'fusion':
			self.cross_attention 	= attention.FusionCrossAttention(dim_a=49, dim_b=49)
		else:
			self.cross_attention 	= attention.Concat()

		self.relation_layer	= RelationLayer('model/third_party/avr_net/weights/best_relation.ckpt')
		self.bna = nn.BatchNorm2d(256)
		self.bna.load_state_dict(self.relation_layer.bna.state_dict())


	def fine_tunning(self):
		self.freeze_relation()
		for param in self.relation_layer.fc.parameters():
			param.requires_grad = True


	def freeze_relation(self):
		for param in self.relation_layer.parameters():
			param.requires_grad = False

		for param in self.bna.parameters():
			param.requires_grad = False


	def unfreeze_relation(self):
		for param in self.relation_layer.parameters():
			param.requires_grad = True

		for param in self.bna.parameters():
			param.requires_grad = True


	def forward(self, video, audio, task):
		# Batch Norm Audio
		B, N, C, H, W = audio.shape

		audio = self.bna(audio.reshape(B*N, C, H, W))
		audio = audio.reshape(B, N, C, H, W)

		# Shaping audio
		audio_a = audio[:, :1, ...]
		audio_b = audio[:, 1:, ...]

		# Shaping video and running self attention
		frames = video.size(dim=1) // 2

		video_a = self.self_attention(video[:, :frames, ...])
		video_b = self.self_attention(video[:, frames:, ...])

		# Cross attention on audio and video
		feats_a = self.cross_attention(video_a, audio_a)
		feats_b = self.cross_attention(video_b, audio_b)

		# Reletion Layer
		feats = torch.cat((feats_a, feats_b), dim=1)
		task = task.t()

		scores = self.relation_layer(feats, task)

		return feats_a, feats_b, scores
