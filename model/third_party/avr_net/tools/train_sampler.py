import torch
import random
from typing import Iterator
from torch.utils.data import Sampler

from .train_dataset import TrainDataset


class TrainSampler(Sampler):
	def __init__(self, dataset: TrainDataset, seed: int=0) -> None:
		self.dataset = dataset
		self.seed = seed
		self.epoch = 0
		self.num_pairs = 20
		self.num_shot = 2
		self.total_size = len(self.dataset.video_ids)


	def __iter__(self) -> Iterator:
		generator = torch.Generator()
		generator.manual_seed(self.seed + self.epoch)

		indices = []
		video_to_entities = self.dataset.video_to_entities

		videos = list(video_to_entities.keys())

		for video, entities_in_video in video_to_entities.items():
			entities = entities_in_video
			random.shuffle(entities)
			entities = entities[:self.num_pairs]

			# If there are not enough entities in the video, fill with entities from other videos.
			if len(entities_in_video) < self.num_pairs:
				other_entities = []
				video_index = videos.index(video)
				other_videos = videos[:video_index] + videos[video_index + 1:]

				for other_video in other_videos:
					other_entities.extend(video_to_entities[other_video])

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
		if len(indices) < self.num_shot: indices *= self.num_shot

		return random.sample(indices, self.num_shot)
