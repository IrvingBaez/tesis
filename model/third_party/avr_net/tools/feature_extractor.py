import torch.nn as nn

from model.third_party.avr_net.models.audio_encoder import AudioEncoder
from model.third_party.avr_net.models.video_encoder import VideoEncoder


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


	def forward(self, audio, video):
		"""
		batch['frames'] is a list of tensors containing the data for every utterance frame.
		batch['audio'] is a tensor containing the data for the audio in all utternaces.
		"""
		# batch['audio'] = batch['audio'].to(self._current_device())
		# batch['frames'] = batch['frames'].to(self._current_device())

		features = (
			self._audio_encoder(audio),
			self._video_encoder(video)
		)

		return features
