import random

import torch
import torch.nn as nn

from model.third_party.avr_net.models.audio_encoder import AudioEncoder
from model.third_party.avr_net.models.video_encoder import VideoEncoder


class FeatureExtractor(nn.Module):
	def __init__(self, mode='pick_first'):
		super().__init__()

		self._audio_encoder	= AudioEncoder('model/third_party/avr_net/weights/backbone_audio.pth')
		self._video_encoder	= VideoEncoder('model/third_party/avr_net/weights/backbone_faces.pth')

		modes = ['pick_first', 'pick_random', 'keep_all', 'average']
		assert mode in modes, f"mode should be in {modes}, not {mode}"
		self.mode = mode

		self._audio_encoder.eval()
		for param in self._audio_encoder.parameters():
			param.requires_grad = False

		self._video_encoder.eval()
		for param in self._video_encoder.parameters():
			param.requires_grad = False

		random.seed('CleoStoat')


	def forward(self, audio, video):
		"""
		batch['frames'] is a list of tensors containing the data for every utterance frame.
		batch['audio'] is a tensor containing the data for the audio in all utternaces.
		"""
		feat_audio = self._audio_encoder(audio)
		B, F, C, _, W, H = video.shape

		if self.mode == 'keep_all':
			feat_video = self._video_encoder(video)
		elif self.mode == 'pick_first':
			feat_video = self._first_video_feature(video)
		elif self.mode == 'pick_random':
			feat_video = self._random_video_feature(video)
		elif self.mode == 'average':
			feat_video = self._average_video_features(video)
		else:
			raise ValueError(f"Unrecognized mode value '{self.mode}'")

		return feat_audio, feat_video


	def remove_zero_padding(self, video):
		B, F, C, _, W, H = video.shape
		no_padding = []

		for idx in range(B):
			frames = []
			last_empty = False

			for frame_idx in range(F):
				next_frame = video[idx:idx+1, frame_idx:frame_idx+1, ...]
				frames.append(next_frame)

				if (next_frame == -1).all().item():
					last_empty = True
					break

			if last_empty and len(frames) > 1: frames.pop()

			frames = torch.cat(frames, dim=1)
			no_padding.append(frames)

		return no_padding


	def _first_video_feature(self, video):
		frames = video[:, 0:1, ...]
		features = self._video_encoder(frames)

		return features


	def _random_video_feature(self, video):
		video_no_padding = self.remove_zero_padding(video)
		frames = []

		for utterance in video_no_padding:
			max_idx = utterance.shape[1] - 1
			frame_idx = random.randint(0, max_idx)

			frames.append(utterance[0:1, frame_idx:frame_idx+1, ...])

		frames = torch.cat(frames)
		features = self._video_encoder(frames)

		return features


	def _average_video_features(self, video):
		video_no_padding = self.remove_zero_padding(video)
		result = []

		for utterance in video_no_padding:
			feat_utterance = self._video_encoder(utterance)
			average = torch.mean(feat_utterance, dim=1)
			average = average.reshape(1, 1, 512, 7, 7)

			result.append(average)

		result = torch.cat(result)
		return result
