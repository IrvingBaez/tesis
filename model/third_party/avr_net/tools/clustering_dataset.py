import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm
from collections import OrderedDict
from bisect import bisect_right
import math
from glob import glob
import random
import os


class ClusteringDataset(Dataset):
	def __init__(self, features_path, disable_pb=False, video_proportion=1.0, cache_size=1):
		super().__init__()

		self.features_path = features_path
		self.video_ids = sorted([os.path.basename(filepath).split('.')[0] for filepath in glob(f'{self.features_path}/*.*')])

		assert 0.0 < video_proportion <= 1.0, 'Video proportion must be in the interval (0,1]'

		if video_proportion < 1.0:
			random.seed('CleoStoat')
			new_list_size = int(len(self.video_ids) * video_proportion)
			self.video_ids = random.sample(self.video_ids, new_list_size)

		self.disable_pb = disable_pb

		self.utterance_counts = OrderedDict()
		self._video_ranges = [0]

		self.cache_size = cache_size
		self.cache = OrderedDict()
		self.cache_hits = 0.0
		self.cache_misses = 0.0

		for video_id in tqdm(self.video_ids, desc='Indexing clustering dataset', disable=self.disable_pb):
			utterances = self._dict_features_to_items(video_id)

			self.utterance_counts[video_id] = len(utterances)
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
			print(self.video_ids)
			print(f'{index:03d} => Vid {list(self.utterance_counts.keys()).index(video_id)}: {video_id} fn({video_size},{inner_index}) = ({i:03d}, {j:03d})')
			print('error')

		return item


	def _inner_index_to_coords(self, n, k):
		i = int((2 * n - 1 - math.sqrt((2 * n - 1)**2 - 8 * k)) // 2)
		k_prime = k - i * (2 * n - i - 1) // 2
		j = i + 1 + k_prime

		return i, j


	def _pairs_in_video(self, video_id):
		n = self.utterance_counts[video_id]

		return ((n - 1) * n) // 2


	def starts(self):
		return self.starts


	def ends(self):
		return self.ends


	def _hit_cache(self, video_id):
		if video_id not in self.cache.keys():
			self.cache_misses += 1.0

			# Make space for new data if necesary
			if len(self.cache) >= self.cache_size:
				self.cache.popitem(last=False)

			# Load new data
			self.cache[video_id] = self._dict_features_to_items(video_id)
		else:
			self.cache_hits += 1.0

		# print(f'Hit/miss ratio = {(self.cache_hits/self.cache_misses):.2f} for cache of size {len(self.cache)} and {len(self.video_ids)} videos')

		return self.cache[video_id]


	def _retrive_item(self, video_id, i, j):
		utterances = self._hit_cache(video_id)

		video_features = torch.cat((utterances[i]['video_features'], utterances[j]['video_features']))
		audio_features = torch.cat((utterances[i]['audio_features'], utterances[j]['audio_features']))

		visible_a = int(utterances[i]['visible'])
		visible_b = int(utterances[j]['visible'])

		task_1 = 2 * visible_a + visible_b
		task_2 = 2 * visible_b + visible_a

		item = {
			# 'video_id': video_id,
			# 'index_a': torch.LongTensor([i]),
			# 'index_b': torch.LongTensor([j]),
			'video': video_features,
			'audio': audio_features,
			'task_full': [task_1, task_2],
			'target': torch.FloatTensor([utterances[i]['target'] == utterances[j]['target']])
		}

		return item


	def _dict_features_to_items(self, video_id):
		features = torch.load(f'{self.features_path}/{video_id}.pckl')

		filtered = {
			'feat_video': features['feat_video'],
			'feat_audio': features['feat_audio'],
			'target': 		features['targets'],
			'visible': 		features['visible'],
			'start': 			features['start'],
			'end': 				features['end']
		}

		video_features = []
		for feat_video, feat_audio, target, visible, start, end in zip(
			filtered['feat_video'], filtered['feat_audio'], filtered['target'], filtered['visible'], filtered['start'], filtered['end']
		):
			video_features.append({
				'video_features':	feat_video,
				'audio_features':	feat_audio,
				'visible':				visible,
				'target':					target,
				# 'start':					start,
				# 'end':						end
			})

		return video_features