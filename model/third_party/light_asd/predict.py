import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import time, os, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, math, python_speech_features, logging

from math import floor
from tqdm import tqdm
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from tqdm.contrib.concurrent import process_map
from scenedetect import open_video, SceneManager, ContentDetector, StatsManager

from model.util import argparse_helper
from .model.faceDetector.s3fd import S3FD
from .ASD import ASD

warnings.filterwarnings("ignore")


def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	logging.getLogger('pyscenedetect').setLevel(logging.ERROR)

	video = open_video(args.videoFilePath)
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	sceneManager.detect_scenes(video)

	sceneList = sceneManager.get_scene_list()
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')

	if sceneList == []:
		sceneList = [(video.base_timecode, video.position)]

	save_pckl(savePath, sceneList)
	log(f'{len(sceneList)} scenes detected')

	return sceneList


def inference_video(args):
	# GPU: Face detection, output is the list of lists of dicts, contains the frame number, face location and confidence score
	detector = S3FD(device='cuda')
	frames = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	frames.sort()
	detections = []

	inference_tasks = []
	for index, frame in tqdm(enumerate(frames), total=len(frames), desc='1: Detecting faces', leave=False):
		inference_tasks.append((index, frame, args.face_det_scale, detector))

	detections = process_map(detect_faces, inference_tasks, max_workers=8, chunksize=1, desc=f'Detecting faces', leave=False)

	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	save_pckl(savePath, detections)

	return detections


def detect_faces(data):
	index, frame, detection_scale, detector = data

	image = cv2.imread(frame)
	imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	bboxes = detector.detect_faces(imageNumpy, conf_th=0.9, scales=[detection_scale])

	results = []
	for bbox in bboxes:
		results.append({'frame':index, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

	return results


def bb_intersection_over_union(boxA, boxB):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou


def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres = 0.5     # Minimum IOU between consecutive face detections
	tracks = []

	while True:
		track = []

		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.num_failed_det:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
				else:
					break

		if track == []:
			break

		if len(track) > args.min_track:
			frameNum    = numpy.array([face['frame'] for face in track])
			bboxes      = numpy.array([numpy.array(face['bbox']) for face in track])
			frameI      = numpy.arange(frameNum[0], frameNum[-1]+1)
			bboxesI     = []

			for ij in range(0, 4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))

			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.min_face_size:
				tracks.append({'frame':frameI,'bbox':bboxesI})

	return tracks


def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	frames = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	frames.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), args.fps, (224,224))
	detections = {'x':[], 'y':[], 's':[]}

	for bbox in track['bbox']:
		# S: half lenght of longest rectangle side.
		detections['s'].append(max((bbox[3]-bbox[1]), (bbox[2]-bbox[0]))/2)
		detections['y'].append((bbox[1]+bbox[3])/2) # crop center x
		detections['x'].append((bbox[0]+bbox[2])/2) # crop center y

	detections['s'] = signal.medfilt(detections['s'], kernel_size=13)  # Smooth detections
	detections['x'] = signal.medfilt(detections['x'], kernel_size=13)
	detections['y'] = signal.medfilt(detections['y'], kernel_size=13)

	for frame_index, frame in enumerate(track['frame']):
		crop_scale  = args.crop_scale
		box_size  = detections['s'][frame_index]   # Detection box size
		box_padding = int(box_size * (1 + 2 * crop_scale))  # Pad videos by this amount

		image = cv2.imread(frames[frame])
		frame = numpy.pad(image, ((box_padding, box_padding), (box_padding, box_padding), (0, 0)), 'constant', constant_values=(110, 110))
		my = detections['y'][frame_index] + box_padding  # BBox center Y
		mx = detections['x'][frame_index] + box_padding  # BBox center X

		y_1 = int(my - box_size)
		y_2 = int(my + box_size * (1 + 2 * box_size))
		x_1 = int(mx - box_size * (1 + crop_scale))
		x_2 = int(mx + box_size * (1 + crop_scale))

		face = frame[y_1:y_2, x_1:x_2]

		vOut.write(cv2.resize(face, (224, 224)))

	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / args.fps
	audioEnd    = (track['frame'][-1]+1) / args.fps
	vOut.release()

	command = f"ffmpeg -y -i {args.audioFilePath} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {args.num_loader_treads} -ss {audioStart} -to {audioEnd} {audioTmp} -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)

	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
				(cropFile, audioTmp, args.num_loader_treads, cropFile)) # Combine audio and video file
	subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')

	return {'track':track, 'proc_track':detections}


def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained model
	detector = ASD()
	detector.loadParameters(args.pretrained_model)
	# print(f"Model {args.pretrained_model} loaded from previous state!")
	detector.eval()
	allScores = []

	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm(files, total = len(files), desc='3: Evaluating network', leave=False):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []

		while video.isOpened():
			ret, frames = video.read()

			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break

		video.release()

		videoFeature = numpy.array(videoFeature)
		length = 4 * min(floor(audioFeature.shape[0] / 4), videoFeature.shape[0]) / 100
		# length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use model

		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []

			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[(i * duration * 100) : ((i+1) * duration * 100),:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[(i * duration * 25) : ((i+1) * duration * 25),:,:]).unsqueeze(0).cuda()
					embedA = detector.model.forward_audio_frontend(inputA)
					embedV = detector.model.forward_visual_frontend(inputV)
					out = detector.model.forward_audio_visual_backend(embedA, embedV)
					score = detector.lossAV.forward(out, labels = None)
					scores.extend(score)

			allScore.append(scores)

		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)

	return allScores


def build_csv(tracks, scores, args):
	video = cv2.VideoCapture(f'{args.pyaviPath}/video.avi')
	height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
	width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

	file_id = args.csv_path.split('/')[-1].split('.')[0]
	segment = int(file_id.split('_')[-1])

	offset = 600 + 300 * segment

	lines = []
	spk_count = 0

	for start in range(0, 300, 60):
		end = start + 60
		section_id = f'{file_id}_{(start+offset):04d}_{(end+offset):04d}'

		for track_index, track in enumerate(tracks):
			if not start <= (track['track']['frame'][0] / args.fps) < end: continue
			entity_id = f'{section_id}:{track_index}'

			track_scores = scores[track_index]

			for frame_index, frame in enumerate(track['track']['frame']):
				bbox = track['track']['bbox'][frame_index]
				score = track_scores[max(frame_index - 2, 0): min(frame_index + 3, len(track_scores) - 1)] # average smoothing
				score = numpy.mean(score)

				# Video id
				line = [file_id]

				# Frame timestamp
				time_stamp = (frame / args.fps) + offset
				time_stamp = round(time_stamp, 6)
				line.append(time_stamp)

				# Bounding box coordinates
				line.append(bbox[0] / width)
				line.append(bbox[1] / height)
				line.append(bbox[2] / width)
				line.append(bbox[3] / height)

				# Speach detection
				line.append('SPEAKING_AUDIBLE' if score > 0 else 'NOT_SPEAKING')

				# Entity id
				line.append(entity_id)

				# Speaker id
				spk_id = 'NA'
				if score > 0:
					spk_id = f'spk{spk_count:02d}'
					spk_count += 1

				line.append(spk_id)

				lines.append(','.join(map(str, line)) + '\n')

	with open(args.csv_path, 'w') as csv:
		for line in lines:
			csv.write(line)


def report_save(data, path):
	log(f"Saved the {data} in {path}")


def log(message):
	global lines_out

	if verbose:
		lines_out += 1
		timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
		print(f'{timestamp}: {message}')


def save_pckl(path, data):
	with open(path, 'wb') as file:
		pickle.dump(data, file)


def predict(args):
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```

	if os.path.exists(args.savePath):
		rmtree(args.savePath)

	os.makedirs(args.pyaviPath, 		exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyframesPath, 	exist_ok = True) # Save all the video frames
	os.makedirs(args.pyworkPath, 		exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(args.pycropPath, 		exist_ok = True) # Save the detected face clips (audio+video) in this process

	# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	duration_args = f'-ss {args.start} -to {args.start + args.duration}'
	if args.duration == 0:
		duration_args = ''

	# Extract video
	command = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads {args.num_loader_treads} {duration_args} -async 1 -r {args.fps} {args.videoFilePath} -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)
	report_save('video', args.videoFilePath)

	# Extract audio
	command = f"ffmpeg -y -i {args.videoFilePath} -qscale:a 0 -ac 1 -vn -threads {args.num_loader_treads} -ar 16000 {args.audioFilePath} -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)
	report_save('audio', args.audioFilePath)

	# Extract the video frames
	command = f"ffmpeg -y -i {args.videoFilePath} -qscale:v 2 -threads {args.num_loader_treads} -f image2 {args.pyframesPath}/%06d.jpg -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)
	report_save('frames', args.pyframesPath)

	# Scene detection for the video framesr
	scene = scene_detect(args)
	report_save('scene', args.pyworkPath)

	# Face detection for the video frames
	faces = inference_video(args)
	report_save('faces', args.pyworkPath)

	# Face tracking
	allTracks = []
	for shot in scene:
		frame_count = shot[1].frame_num - shot[0].frame_num

		if frame_count < args.min_track:
			continue

		allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	log(f"Face track detected {len(allTracks)} tracks")

	# Face clips cropping
	vidTracks = []
	for index, track in tqdm(enumerate(allTracks), total = len(allTracks), desc='2: Clipping faces', leave=False):
		vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%index)))

	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	save_pckl(savePath, vidTracks)
	report_save('face crops', args.pycropPath)

	# Active Speaker Detection
	files = glob.glob("%s/*.avi"%args.pycropPath)
	files.sort()
	scores = evaluate_network(files, args)

	savePath = os.path.join(args.pyworkPath, 'scores.pckl')
	save_pckl(savePath, scores)
	report_save('scores', args.pyworkPath)

	log('Rendering track csv')
	build_csv(vidTracks, scores, args)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--video_name',					type=str, default="col",   help='Demo video name')
	parser.add_argument('--video_folder',				type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')
	parser.add_argument('--pretrained_model',		type=str, default="model/third_party/light_asd/weight/pretrain_AVA_CVPR.model",   help='Path for the pretrained model')
	parser.add_argument('--csv_path',						type=str, help='Path to create predictions file')
	parser.add_argument('--verbose',						action='store_true', help='Print progress and process')

	parser.add_argument('--num_loader_treads',	type=int,   default=10,   help='Number of workers')
	parser.add_argument('--face_det_scale',			type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
	parser.add_argument('--min_track',					type=int,   default=10,   help='Number of min frames for each shot')
	parser.add_argument('--num_failed_det',			type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
	parser.add_argument('--min_face_size',			type=int,   default=1,    help='Minimum face size in pixels')
	parser.add_argument('--crop_scale',					type=float, default=0.40, help='Scale bounding box')

	parser.add_argument('--start',							type=int, default=0,   help='The start time of the video')
	parser.add_argument('--duration',						type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

	args = argparse_helper(parser, **kwargs)

	global verbose
	verbose = args.verbose
	global lines_out
	lines_out = 0

	args.videoPath 			= glob.glob(os.path.join(args.video_folder, args.video_name + '.*'))[0]
	args.savePath 			= os.path.join(args.video_folder, args.video_name)
	args.pyaviPath 			= os.path.join(args.savePath, 'pyavi')
	args.pyframesPath 	= os.path.join(args.savePath, 'pyframes')
	args.pyworkPath 		= os.path.join(args.savePath, 'pywork')
	args.pycropPath 		= os.path.join(args.savePath, 'pycrop')
	args.videoFilePath 	= os.path.join(args.pyaviPath, 'video.avi')
	args.audioFilePath 	= os.path.join(args.pyaviPath, 'audio.wav')

	cam = cv2.VideoCapture(args.videoPath)
	args.fps = cam.get(cv2.CAP_PROP_FPS)
	cam.release()

	if args.csv_path == None:
		args.csv_path = args.savePath + '/csv'

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	predict(args)


if __name__ == '__main__':
	args = initialize_arguments()
	predict(args)
