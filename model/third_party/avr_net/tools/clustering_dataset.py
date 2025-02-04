import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from collections import OrderedDict
from bisect import bisect_right
import math


class ClusteringDataset(Dataset):
	def __init__(self, features_path, disable_pb=False):
		super().__init__()

		self.features = torch.load(features_path)
		self.disable_pb = disable_pb

		self.items = []
		self.utterance_counts = OrderedDict()
		self.start = {}
		self.end = {}
		self._video_ranges = [0]
		self.features_by_video = {}

		self.video_ids = sorted(list(set(self.features['video'])))
		for video_id in tqdm(self.video_ids, desc='Loading clustering dataset', leave=False, disable=self.disable_pb):
			utterances = self._features_by_video(video_id)

			self.utterance_counts[video_id] = len(utterances)
			self.start[video_id]	= np.array([utterance['start'] for utterance in utterances])
			self.end[video_id] 		= np.array([utterance['end'] for utterance in utterances])
			self.features_by_video[video_id] = self._features_by_video(video_id)
			self._video_ranges.append(self._video_ranges[-1] + self._pairs_in_video(video_id))

		self._video_ranges.pop()


	def __len__(self):
		total = 0

		# Counts elements in the upper diagonal of the square matrix of side of size = utterances in video
		for video_id in self.video_ids:
			total += self._pairs_in_video(video_id)

		return total


	def __getitem__(self, index):
		# Find corresponding video
		video_index = bisect_right(self._video_ranges, index) - 1

		cumulative 	= self._video_ranges[video_index]
		video_id 		= list(self.utterance_counts.keys())[video_index]
		video_size 	= list(self.utterance_counts.values())[video_index]

		# Index of element in video
		inner_index = index - cumulative

		# Convert index to coordinates of element in matrix
		i, j = self._inner_index_to_coords(video_size, inner_index)

		try:
			item = self._retrive_item(video_id, i, j)
		except Exception as e:
			print(repr(e))
			print(f'{index:03d} => Vid {list(self.utterance_counts.keys()).index(video_id)}: {video_id} fn({video_size},{inner_index}) = ({i:03d}, {j:03d})')
			print('error')

		return item


	def _inner_index_to_coords(self, n, k):
		i = int((2 * n - 1 - math.sqrt((2 * n - 1)**2 - 8 * k)) // 2)
		k_prime = k - i * (2 * n - i - 1) // 2
		j = i + 1 + k_prime

		return i, j


	def _pairs_in_video(self, video_id):
		n = self.count_utterances(video_id)

		return ((n - 1) * n) // 2



	def count_utterances(self, video_id):
		return self.utterance_counts[video_id]


	def starts(self):
		return self.starts


	def ends(self):
		return self.ends


	def _retrive_item(self, video_id, i, j):
		utterances = self.features_by_video[video_id]
		self.utterance_counts[video_id] = len(utterances)

		video_features = torch.cat((utterances[i]['video_features'], utterances[j]['video_features']))
		audio_features = torch.cat((utterances[i]['audio_features'], utterances[j]['audio_features']))

		visible_a = int(utterances[i]['visible'])
		visible_b = int(utterances[j]['visible'])

		task_1 = 2 * visible_a + visible_b
		task_2 = 2 * visible_b + visible_a

		item = {
			'video_id': video_id,
			'index_a': torch.LongTensor([i]),
			'index_b': torch.LongTensor([j]),
			'video': video_features,
			'audio': audio_features,
			'task_full': [task_1, task_2],
			'target': torch.FloatTensor([utterances[i]['target'] == utterances[j]['target']])
		}

		return item


	def _features_by_video(self, video_id):
		indices = torch.LongTensor([index for index, value in enumerate(self.features['video']) if value == video_id])

		self.features['targets'] = self.features['targets']

		filtered = {
			'feat_video': torch.index_select(self.features['feat_video'], 0, indices),
			'feat_audio': torch.index_select(self.features['feat_audio'], 0, indices),
			'target': 		torch.index_select(self.features['targets'], 		0, indices),
			'visible': 		[self.features['visible'][index]	for index in indices],
			'start': 			[self.features['start'][index]		for index in indices],
			'end': 				[self.features['end'][index]			for index in indices],
		}

		video_features = []
		for feat_video, feat_audio, target, visible, start, end in zip(
			filtered['feat_video'], filtered['feat_audio'], filtered['target'], filtered['visible'], filtered['start'], filtered['end']
		):
			video_features.append({
				'video_features':	feat_video,
				'audio_features':	feat_audio.cpu(),
				'visible':				visible,
				'target':					target.cpu(),
				'start':					start,
				'end':						end
			})

		return video_features