from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
import torch
import sys
import os
from model.util import save_data


class TrainDataset(Dataset):
	def __init__(self, features_path, video_ids, video_proportion=1.0, disable_pb=False):
		super().__init__()

		self.features_path = features_path
		self.video_ids = sorted(video_ids)
		self.disable_pb = disable_pb
		self._video_ranges = {}

		index_path = f'{features_path}/video_index.pckl'

		if not os.path.exists(index_path):
			self.size = 0
			for video_id in tqdm(self.video_ids, desc='Indexing videos', disable=self.disable_pb):
				self._loaded_features = torch.load(f'{features_path}/{video_id}.pckl')
				feature_count = len(self._loaded_features)

				self._video_ranges[video_id] = (self.size, self.size + feature_count - 1)
				self.size += feature_count
				del self._loaded_features

			save_data((self._video_ranges, self.size), index_path)
		else:
			self._video_ranges, self.size = torch.load(index_path)

		self._loaded_features = torch.load(f'{features_path}/{self.video_ids[0]}.pckl')
		self._loaded_video_id = self.video_ids[0]
		self.size = int(self.size * video_proportion)


	def __len__(self):
		return self.size


	def __getitem__(self, index):
		video_id, feature_index = self.index_to_video_id(index)
		self.load_video_data(video_id)

		return self._loaded_features[feature_index]


	def index_to_video_id(self, index):
		for video_id, (start, end) in self._video_ranges.items():
			if start <= index <= end:
				return video_id, index - start

		raise ValueError(f'Datum with index {index} not found in TrainDataset.')


	def load_video_data(self, video_id):
		if self._loaded_video_id == video_id: return

		self._loaded_features = torch.load(f'{self.features_path}/{video_id}.pckl')
		self._loaded_video_id = video_id
