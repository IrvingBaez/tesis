import torch
import numpy as np


class AHC_Cluster():
	def __init__(self, threshold):
		super().__init__()
		self.threshold = threshold


	def fit_predict(self, similarity):
		if isinstance(similarity, torch.Tensor):
			similarity = similarity.cpu().numpy()

		dist = -similarity

		return self._ahc(dist, threshold=self.threshold)


	def _ahc(self, dist, threshold=0.3):
		dist[np.diag_indices_from(dist)] = np.inf
		clusters = [[i] for i in range(len(dist))]

		while True:
			index_a, index_b = np.sort(np.unravel_index(dist.argmin(), dist.shape))

			if dist[index_a, index_b] > -threshold:
				break

			dist[:, index_a] = dist[index_a,:] = (
				dist[index_a,:] * len(clusters[index_a]) + dist[index_b,:] * len(clusters[index_b])
			) / (
				len(clusters[index_a]) + len(clusters[index_b])
			)


			dist[:, index_b] = dist[index_b,:] = np.inf
			clusters[index_a].extend(clusters[index_b])
			clusters[index_b] = None

		labs= np.empty(len(dist), dtype=int)
		for index, cluster in enumerate([cluster for cluster in clusters if cluster]):
			labs[cluster] = index

		return labs
