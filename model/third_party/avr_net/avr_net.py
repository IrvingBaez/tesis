import torch.nn as nn

from model.third_party.avr_net.models.relation_layer import RelationLayer
from model.third_party.avr_net.models.audio_encoder import AudioEncoder
from model.third_party.avr_net.models.video_encoder import VideoEncoder


class AVRNET(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config


	def build(self, relation_layer='original'):
		self.audio_encoder	= AudioEncoder(self.config['audio']['init_weight'])
		self.video_encoder	= VideoEncoder(self.config['video']['init_weight'])

		self.relation_layer	= RelationLayer()


	def train(self, mode=True):
		super().train(mode)
		if self.config['audio']['fix_layers']:
			for module in self.audio_encoder.modules():
				if isinstance(module, nn.BatchNorm2d):
					module.eval()


		if self.config['video']['fix_layers']:
			for module in self.video_encoder.modules():
				if isinstance(module, nn.BatchNorm2d):
					module.eval()
					module.weight.requires_grad = False
					module.bias.requires_grad = False


	# TODO: This is terrible code, `forward` should't manage or split like this.
	def forward(self, batch, exec:str=None):
		if exec == 'train':
			return self._forward_train(batch)
		else:
			return self._forward_predict(batch, exec)


	def _forward_train(self, batch):
		feat_audio = self.audio_encoder(batch)
		feat_video = self.video_encoder(batch)

		scores, targets = self.relation_layer(feat_video, feat_audio, batch['visible'], batch['targets'])

		return {'scores': scores, 'targets': targets}


	def _forward_predict(self, batch, exec):
		output = {}

		if exec == 'extraction':
			feat_audio = self.audio_encoder(batch)
			feat_video = self.video_encoder(batch)

			output['feat_audio']	= feat_audio
			output['feat_video']	= feat_video
			output['video']				= batch['meta']['video']
			output['start']				= batch['meta']['start']
			output['end']					= batch['meta']['end']
			output['trackid']			= batch['meta']['trackid']
			output['visible']			= batch['visible']
			output['losses']			= {}
		elif exec == 'relation':
			output['scores']			= self.relation_layer(batch['video'], batch['audio'], batch['task_full'])

		return output
