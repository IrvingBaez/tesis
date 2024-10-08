import argparse
import numpy as np
import os
import torch

from model.util import argparse_helper
from model.third_party.avr_net.tools.ahc_cluster import AHC_Cluster


def write_rttms(args):
	os.makedirs(f'{args.sys_path}/predictions', exist_ok=True)

	threshold = 0.14
	cluster = AHC_Cluster(threshold)
	rttm_list = []

	similarity_data = torch.load(args.similarities_path)

	for video_id in similarity_data['similarities']:
		similarity = similarity_data['similarities'][video_id]
		starts = similarity_data['starts'][video_id]
		ends = similarity_data['ends'][video_id]
		labels = cluster.fit_predict(similarity)

		starts, ends, labels = merge_frames(starts, ends, labels)

		lines = []
		for label, start, end in zip(labels, starts, ends):
			if start < 0: start = 0
			if end - start < 0.01: continue

			lines.append(f'SPEAKER {video_id} 1 {start:010.6f} {(end-start):010.6f} <NA> <NA> spk{label:02d} <NA> <NA>\n')

		pred_path = f'{args.sys_path}/predictions/{video_id}.rttm'
		rttm_list.append(pred_path + '\n')

		with open(pred_path, 'w') as file:
			file.writelines(lines)

	with open(f'{args.sys_path}/{args.data_type}.out', 'w') as file:
		file.writelines(rttm_list)


def merge_frames(starts, ends, labels):
	# Sort the segments by their start times
	sorted_indices = np.argsort(starts)
	starts, ends, labels = starts[sorted_indices], ends[sorted_indices], labels[sorted_indices]

	# Identify adjacent or overlapping segments with the same label
	adjacent_or_overlapping = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
	different_labels_or_separated = np.logical_or(~adjacent_or_overlapping, labels[1:] != labels[:-1])
	split_indices = np.nonzero(different_labels_or_separated)[0]

	# Merge the segments based on the identified split points
	merged_starts = starts[np.r_[0, split_indices + 1]]
	merged_ends = ends[np.r_[split_indices, -1]]
	merged_labels = labels[np.r_[0, split_indices + 1]]

	# Adjust for overlapping segments by averaging the overlap points
	overlapping_indices = np.nonzero(merged_starts[1:] < merged_ends[:-1])[0]
	merged_ends[overlapping_indices] = merged_starts[overlapping_indices + 1] = (
		merged_ends[overlapping_indices] + merged_starts[overlapping_indices + 1]
	) / 2.0

	return merged_starts, merged_ends, merged_labels


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--data_type',					type=str,	help='Type of data being processed, test, val or train')
	parser.add_argument('--sys_path',						type=str,	help='Path to the folder where to save all the system outputs')
	parser.add_argument('--similarities_path',	type=str,	help='Path to the file with similarity data')

	args = argparse_helper(parser, **kwargs)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	write_rttms(args)


if __name__ == '__main__':
	args = initialize_arguments()
	write_rttms(args)