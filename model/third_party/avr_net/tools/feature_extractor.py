import torch
import torch.nn as nn

from model.third_party.avr_net.models.audio_encoder import AudioEncoder
from model.third_party.avr_net.models.video_encoder import VideoEncoder
import model.third_party.avr_net.tools.attention as attention


class FeatureExtractor(nn.Module):
	def __init__(self):
		super().__init__()

		self._audio_encoder	= AudioEncoder('model/third_party/avr_net/weights/backbone_audio.pth')
		self._video_encoder	= VideoEncoder('model/third_party/avr_net/weights/backbone_faces.pth')

		self._audio_encoder.eval()
		for param in self._audio_encoder.parameters():
			param.requires_grad = False

		self._video_encoder.eval()
		for param in self._video_encoder.parameters():
			param.requires_grad = False

		self._frames_attention = attention.PickFirst()
		self._cross_attention = attention.Identity()


	def forward(self, batch:dict):
		"""
		batch['frames'] is a list of tensors containing the data for every utterance frame.
		batch['audio'] is a tensors containing the data for the audio in all utternaces.
		"""
		batch['audio'] = batch['audio'].to(self._current_device())

		features = {
			'feat_audio':	self._audio_encoder(batch['audio']),
			'feat_video':	self._encode_video(batch['frames']),
			'video':			batch['meta']['video'],
			'start':			batch['meta']['start'],
			'end':				batch['meta']['end'],
			'trackid':		batch['meta']['trackid'],
			'visible':		batch['visible'].to(self._current_device()),
			'losses':			{}
		}

		if 'targets' in batch.keys():
			features['targets'] = batch['targets'].to(self._current_device())

		return features


	def _encode_video(self, video_batch):
		attended_utterances = []

		for	utterance_frames in video_batch:
			utterance_frames = utterance_frames.to(self._current_device())
			utterance_encoding = self._video_encoder(utterance_frames)
			utterance_encoding = self._frames_attention(utterance_encoding[0])
			utterance_encoding = utterance_encoding.reshape(1, *utterance_encoding.shape)

			attended_utterances.append(utterance_encoding)

		video_encodings = torch.cat(attended_utterances, dim=0)

		return video_encodings


	def _current_device(self):
		return next(self.parameters()).device
