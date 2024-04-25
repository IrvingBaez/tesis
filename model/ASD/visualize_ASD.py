import cv2, argparse
import pandas as pd
from tqdm import tqdm


def visualization(args):
	video = cv2.VideoCapture(args.video_path)

	fps 		= video.get(cv2.CAP_PROP_FPS)
	height 	= int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width 	= int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

	track = quantize_csv(args.csv_path, height, width, fps)
	ground_truth = quantize_csv(args.gt_path, height, width, fps)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	output = cv2.VideoWriter('ASD_visualization.avi', fourcc, fps, (width, height))

	frame_count = 0
	while True:
		ret, frame = video.read()

		if not ret:
			break

		# For the track
		rectangles = track.loc[track['frame_timestamp'] == frame_count]
		for _, rentangle in rectangles.iterrows():
			point_1 = (rentangle['entity_box_x1'], rentangle['entity_box_y1'])
			point_2 = (rentangle['entity_box_x2'], rentangle['entity_box_y2'])
			color = (0, 255, 0) if rentangle['label'] == 'SPEAKING_AUDIBLE' else (0, 0, 255)
			thickness = 2

			cv2.rectangle(frame, point_1, point_2, color, thickness)

		# For Ground Truth
		rectangles = ground_truth.loc[ground_truth['frame_timestamp'] == frame_count]
		for _, rentangle in rectangles.iterrows():
			point_1 = (rentangle['entity_box_x1'], rentangle['entity_box_y1'])
			point_2 = (rentangle['entity_box_x2'], rentangle['entity_box_y2'])
			color = (0, 128, 0) if rentangle['label'] == 'SPEAKING_AUDIBLE' else (0, 0, 128)
			thickness = 2

			cv2.rectangle(frame, point_1, point_2, color, thickness)

		output.write(frame)
		frame_count += 1

	video.release()
	output.release()
	cv2.destroyAllWindows()


def quantize_csv(csv_path, height, width, fps):
	colnames = ['video_id','frame_timestamp','entity_box_x1','entity_box_y1','entity_box_x2','entity_box_y2','label', 'entity_id', 'spkid']

	if csv_path is None:
		return pd.DataFrame(columns=colnames)

	track = pd.read_csv(csv_path, header=None, names=colnames)

	track = track.sort_values('frame_timestamp')

	track['entity_box_x1'] = track['entity_box_x1'].apply(lambda x: int(x * width))
	track['entity_box_y1'] = track['entity_box_y1'].apply(lambda x: int(x * height))
	track['entity_box_x2'] = track['entity_box_x2'].apply(lambda x: int(x * width))
	track['entity_box_y2'] = track['entity_box_y2'].apply(lambda x: int(x * height))

	track['frame_timestamp'] = track['frame_timestamp'].apply(lambda x: int(x * fps))

	return track


def main():
	parser = argparse.ArgumentParser(description = "Arguments for ASD vizualization")
	parser.add_argument('-vp', '--videoPath', 				dest='video_path',  type=str, help='Location of video to process', 					required=True)
	parser.add_argument('-cp', '--csvPath', 					dest='csv_path',  	type=str, help='Location of tracks csv to draw', 				required=True)
	parser.add_argument('-gtp', '--groundTruthPath', 	dest='gt_path',  		type=str, help='Location of ground truth csv to draw', 	default=None)
	parser.add_argument('-op', '--outputPath', 				dest='output_path', type=str, help='Path to write the output video', 				default='')

	args = parser.parse_args()
	visualization(args)


if __name__ == '__main__':
	main()