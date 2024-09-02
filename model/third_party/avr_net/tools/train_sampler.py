import torch
import random
from typing import Iterator
from torch.utils.data import DistributedSampler

from .train_dataset import TrainDataset


class TrainSampler(DistributedSampler):
	def __init__(self, dataset: TrainDataset, num_replicas=None, rank=None, shuffle=True, seed: int=0, num_pairs=20, num_shot=2) -> None:
		super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
		self.dataset = dataset
		self.seed = seed
		self.epoch = 0
		self.num_pairs = num_pairs
		self.num_shot = num_shot
		self.total_size = len(self.dataset.video_ids)
		self.video_to_entities = self.dataset.video_to_entities


	def __iter__(self) -> Iterator:
		generator = torch.Generator()
		generator.manual_seed(self.seed + self.epoch)

		indices = []

		videos = list(self.video_to_entities.keys())

		for video, entities_in_video in self.video_to_entities.items():
			entities = entities_in_video
			random.shuffle(entities)
			entities = entities[:self.num_pairs]

			# If there are not enough entities in the video, fill with entities from other videos.
			if len(entities_in_video) < self.num_pairs:
				other_entities = []
				video_index = videos.index(video)
				other_videos = videos[:video_index] + videos[video_index + 1:]

				for other_video in other_videos:
					other_entities.extend(self.video_to_entities[other_video])

				entities += random.sample(other_entities, (self.num_pairs - len(entities_in_video)))

			# Get 2 pairs corresponding to the same entity.
			samples = [self._sample_same_entity(label) for label in entities]
			indices.append(samples)

		self.total_size = len(indices)

		return iter(indices)


	def __len__(self) -> int:
		return self.total_size


	def _sample_same_entity(self, label):
		indices = self.dataset.entity_to_indices[label]
		if len(indices) < self.num_shot:
			indices *= self.num_shot

		return random.sample(indices, self.num_shot)


	def set_epoch(self, epoch: int):
		super().set_epoch(epoch)
		self.epoch = epoch
