import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
from tqdm.auto import trange

from .clustering_dataset import pair_item


class EpisodicDataset(Dataset):
	"""
	Pairs of the episodes of the original AVR-Net training, as in its official code (github.com/showlab/AVA-AVD:
	FewShotAVASampler in model/mmsc/datasets/samplers/fewshot_sampler.py and divide_set in
	model/mmsc/models/token/encoders/relation.py).

	An episode has num_way speakers with two utterances each, one used as query and the other as support.
	The speakers of a clip are completed with speakers of other videos (never another clip of the same video).
	Every query is compared with every support: num_way^2 pairs, num_way of them positive.

	Only the pairs differ from balanced training: each epoch has as many pairs as the balanced dataset (every
	positive pair plus as many negatives), taken from new episodes, and they are shuffled and batched like any
	other pairs. The episodes of every epoch are generated here from the seed, so a resumed run gets the same ones.
	Use EpochSampler, which picks the pairs of the current epoch.

	pairs is a ClusteringDataset with every pair (not balanced), preloaded. It also gives the train DER.
	"""
	def __init__(self, pairs, epochs, seed, num_way=20, disable_pb=False):
		super().__init__()
		self.pairs = pairs
		self.num_way = num_way

		# {video_id: {speaker label: [utterance indices]}}
		self.speakers = {}
		for video_id in pairs.video_ids:
			groups = defaultdict(list)
			for index, utterance in enumerate(pairs.cache[video_id]):
				groups[int(utterance['target'].flatten()[0])].append(index)
			self.speakers[video_id] = dict(groups)

		self.all_speakers = [(video_id, label) for video_id, groups in self.speakers.items() for label in groups]
		self.video_position = {video_id: position for position, video_id in enumerate(pairs.video_ids)}

		self.episodes_per_epoch = math.ceil(2 * pairs.positive_pairs() / num_way**2)
		self.pairs_per_epoch = self.episodes_per_epoch * num_way**2

		# Every pair: (query video, query utterance, support video, support utterance), as positions in pairs.video_ids
		rng = random.Random(seed)
		episodes = [
			self._episode(rng, pairs.video_ids[index % len(pairs.video_ids)])
			for index in trange(epochs * self.episodes_per_epoch, desc='Generating episodes', disable=disable_pb)
		]
		# (episodes, num_way, 3): video, query, support. int16 keeps the ~70 M pairs of 10 epochs in ~560 MB
		assert max(pairs.utterance_counts.values()) <= np.iinfo(np.int16).max
		shots = np.array(episodes, dtype=np.int16)

		query, support = np.meshgrid(np.arange(num_way), np.arange(num_way), indexing='ij')
		query, support = query.flatten(), support.flatten()
		self.query_video	= shots[:, query, 0].reshape(-1)
		self.query			= shots[:, query, 1].reshape(-1)
		self.support_video	= shots[:, support, 0].reshape(-1)
		self.support		= shots[:, support, 2].reshape(-1)
		self.same_speaker	= np.tile(query == support, len(shots))


	def _episode(self, rng, video_id):
		speakers = [(video_id, label) for label in self.speakers[video_id]]
		rng.shuffle(speakers)
		speakers = speakers[:self.num_way]

		chosen = set(speakers)
		while len(speakers) < self.num_way:
			candidate = rng.choice(self.all_speakers)
			if candidate[0][:11] != video_id[:11] and candidate not in chosen:
				speakers.append(candidate)
				chosen.add(candidate)

		# Query and support of each speaker. With a single utterance, both are the same one, as in the original
		shots = []
		for speaker_video, label in speakers:
			group = self.speakers[speaker_video][label]
			query, support = rng.sample(group, 2) if len(group) > 1 else (group[0], group[0])
			shots.append((self.video_position[speaker_video], query, support))

		return shots


	def __len__(self):
		return len(self.same_speaker)


	def __getitem__(self, index):
		query_video = self.pairs.video_ids[self.query_video[index]]
		support_video = self.pairs.video_ids[self.support_video[index]]
		query, support = int(self.query[index]), int(self.support[index])

		return pair_item(
			query_video, query, support,
			self.pairs.cache[query_video][query], self.pairs.cache[support_video][support], bool(self.same_speaker[index])
		)


class EpochSampler(Sampler):
	"""Shuffled pairs of the current epoch of an EpisodicDataset. Lightning calls set_epoch before every epoch."""
	def __init__(self, dataset, seed):
		self.pairs_per_epoch = dataset.pairs_per_epoch
		self.seed = seed
		self.epoch = 0


	def set_epoch(self, epoch):
		self.epoch = epoch


	def __len__(self):
		return self.pairs_per_epoch


	def __iter__(self):
		generator = torch.Generator().manual_seed(self.seed + self.epoch)
		offset = self.epoch * self.pairs_per_epoch

		yield from (offset + torch.randperm(self.pairs_per_epoch, generator=generator)).tolist()
