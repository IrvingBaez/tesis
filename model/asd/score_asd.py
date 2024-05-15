import numpy as np
import pandas, argparse
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from model.util import argparse_helper, intersection_over_union
from tqdm import tqdm


def load_asd_file(file_path):
	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label', 'entity_id', 'spkid']
	data_frame = pandas.read_csv(file_path, header=None, names=colnames)

	data_frame['label'] = data_frame['label'].apply(lambda label: 1 if label == 'SPEAKING_AUDIBLE' else 0)

	return data_frame


def frame_bbox_and_labels(frame):
	bbox = [
		frame['entity_box_x1'],
		frame['entity_box_y1'],
		frame['entity_box_x2'],
		frame['entity_box_y2']
	]
	label = frame['label']
	entity = frame['entity_id']

	return (bbox, label, entity)


def generate_prediction_sets(dict_gt, dict_pred, unique_entities):
	prediction_set = {entity: [[], []] for entity in unique_entities}
	unique_timestamps = dict_gt['frame_timestamp'].unique()

	for timestamp in tqdm(unique_timestamps, desc='Scoring ASD', leave=False):
		frame_gt = dict_gt[dict_gt['frame_timestamp']==timestamp]
		frame_pred = dict_pred[dict_pred['frame_timestamp'].between(timestamp-0.01, timestamp+0.01)]

		if frame_gt.empty:
			continue

		for _, row_gt in frame_gt.iterrows():
			face_gt, label_gt, id_gt = frame_bbox_and_labels(row_gt)

			# Find the detected face that best aligns to gt, with a minimum of 0.5
			best_iou = 0.5
			best_label = 0
			for _, row_pred in frame_pred.iterrows():
				face_pred, label_pred, _ = frame_bbox_and_labels(row_pred)

				iou = intersection_over_union(face_pred, face_gt, relativeToBoxA=True)
				if iou > best_iou:
					best_label = label_pred

			prediction_set[id_gt][0].append(best_label)
			prediction_set[id_gt][1].append(label_gt)

	return prediction_set


def calculate_average_precision(labels, predictions):
	if np.max(labels) == 0:
		if np.max(predictions) == 0:
			return 1.0

		return 0.0

	return average_precision_score(labels, predictions)


def score_ASD(args):
	dict_gt = load_asd_file(args.gt_path)
	dict_pred = load_asd_file(args.pred_path)
	unique_entities = dict_gt['entity_id'].unique()

	prediction_set = generate_prediction_sets(dict_gt, dict_pred, unique_entities)

	f1_total = 0
	ap_total = 0
	entity_count = len(unique_entities)

	lines = []
	for entity in unique_entities:
		predictions = np.array(prediction_set[entity][0])
		labels = np.array(prediction_set[entity][1])

		f1_current = f1_score(labels, predictions, zero_division=1)
		ap_current = calculate_average_precision(labels, predictions)
		accuracy = accuracy_score(labels, predictions)

		f1_total += f1_current
		ap_total += ap_current

		line = f'{entity}:\tACC:\t{(100 * accuracy):0>6.2f}\tAP:\t{(100 * ap_current):0>6.2f}\tF1:\t{(100 * f1_current):0>6.2f}'
		lines.append(line)

		if args.verbose:
			print(line)

	f1_total /= entity_count
	ap_total /= entity_count

	line = f'\nAverage F1:\t{(100 * f1_total):0>6.2f}\nmAP:\t\t{(100 * ap_total):0>6.2f}'
	lines.append(line)

	if args.save_path is not None:
		with open(args.save_path, 'w') as file:
			file.write('\n'.join(lines))


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--gt_path', 		type=str, required=True, help='Path of ground truth csv')
	parser.add_argument('--pred_path',	type=str, required=True, help='Path of csv predicted by ASD system')
	parser.add_argument('--save_path', 	type=str, help='Path to save the result report')
	parser.add_argument('--verbose', 		action='store_true', help='Print scoring details')

	args = argparse_helper(parser, **kwargs)

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	score_ASD(args)


if __name__ == '__main__':
	args = initialize_arguments()
	score_ASD(args)
