import time, sys, os, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, math, python_speech_features

from tqdm import tqdm
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scenedetect import open_video, SceneManager, ContentDetector, StatsManager
from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")


def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
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
	# GPU: Face detection, output is the list of dicts, contains the frame number, face location and confidence score
	detector = S3FD(device='cuda')
	frames = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	frames.sort()
	dets = []

	for index, frame in tqdm(enumerate(frames), total=len(frames), desc='Detecting faces', leave=False, disable=not verbose):
		image = cv2.imread(frame)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		bboxes = detector.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
			dets[-1].append({'frame':index, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info

	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	save_pckl(savePath, dets)

	return dets


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
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
				else:
					break

		if track == []:
			break

		if len(track) > args.minTrack:
			frameNum    = numpy.array([face['frame'] for face in track])
			bboxes      = numpy.array([numpy.array(face['bbox']) for face in track])
			frameI      = numpy.arange(frameNum[0], frameNum[-1]+1)
			bboxesI     = []

			for ij in range(0, 4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))

			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
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
		crop_scale  = args.cropScale
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

	command = f"ffmpeg -y -i {args.audioFilePath} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {args.nDataLoaderThread} -ss {audioStart} -to {audioEnd} {audioTmp} -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)

	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
				(cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')

	return {'track':track, 'proc_track':detections}


def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained model
	detector = ASD()
	detector.loadParameters(args.pretrainModel)
	# print(f"Model {args.pretrainModel} loaded from previous state!")
	detector.eval()
	allScores = []

	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm(files, total = len(files), desc='Evaluating network', leave=False, disable=not verbose):
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
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
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


def visualization(tracks, scores, args):
	# CPU: visulize the result for video format
	files = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	files.sort()
	faces = [[] for i in range(len(files))]

	for track_index, track in enumerate(tracks):
		score = scores[track_index]

		for frame_index, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(frame_index - 2, 0): min(frame_index + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			faces[frame].append({'track':track_index, 'score':float(s),'s':track['proc_track']['s'][frame_index], 'x':track['proc_track']['x'][frame_index], 'y':track['proc_track']['y'][frame_index]})

	firstImage = cv2.imread(files[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), args.fps, (fw,fh))
	colorDict = {0: 0, 1: 255}

	for file_index, fname in tqdm(enumerate(files), total = len(files), desc='Visualizing', leave=False, disable=not verbose):
		image = cv2.imread(fname)

		for face in faces[file_index]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)

		vOut.write(image)
	vOut.release()

	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi')))
	subprocess.call(command, shell=True, stdout=None)


def build_csv(tracks, scores, args):
	os.makedirs(args.csvPath, exist_ok = True)
	video = cv2.VideoCapture(f'{args.pyaviPath}/video.avi')
	height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
	width = video.get(cv2.CAP_PROP_FRAME_WIDTH)

	file_id = '_'.join(args.videoName.split('_')[:-1])

	lines = []
	entity_ids = []

	for track_index, track in enumerate(tracks):
		track_scores = scores[track_index]
		track_start = f'{int(track['track']['frame'][0] / args.fps):04d}'
		track_end = f'{int(track['track']['frame'][-1] / args.fps):04d}'

		for frame_index, frame in enumerate(track['track']['frame']):
			bbox = track['track']['bbox'][frame_index]
			score = track_scores[max(frame_index - 2, 0): min(frame_index + 3, len(track_scores) - 1)] # average smoothing
			score = numpy.mean(score)

			# Video id
			line = [file_id]

			# Frame timestamp
			time_stamp = frame / args.fps
			time_stamp = math.floor(time_stamp * 100) / 100
			line.append(time_stamp)

			# Bounding box coordinates
			line.append(bbox[0] / width)
			line.append(bbox[1] / height)
			line.append(bbox[2] / width)
			line.append(bbox[3] / height)

			# Speach detection
			line.append('SPEAKING_AUDIBLE' if score > 0 else 'NOT_SPEAKING')

			lines.append(','.join(map(str, line)) + '\n')

	with open(f'{args.csvPath}/{args.videoName}.csv', 'w') as csv:
		for line in lines:
			csv.write(line)


def initialize_args():
	parser = argparse.ArgumentParser(description = "Light ASD prediction")

	parser.add_argument('--videoName',             type=str, default="col",   help='Demo video name')
	parser.add_argument('--videoFolder',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')
	parser.add_argument('--pretrainModel',         type=str, default="third_party/Light_ASD/weight/pretrain_AVA_CVPR.model",   help='Path for the pretrained model')
	parser.add_argument('--csvPath',             	 type=str, help='Path to create predictions file')
	parser.add_argument('--verbose',               action='store_true', help='Print progress and process')
	parser.add_argument('--visualize',             action='store_true', help='Create video with bounding boxes')

	parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
	parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
	parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
	parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
	parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
	parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

	parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
	parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

	args = parser.parse_args()

	global verbose
	verbose = args.verbose
	global lines_out
	lines_out = 0

	args.videoPath 			= glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.savePath 			= os.path.join(args.videoFolder, args.videoName)
	args.pyaviPath 			= os.path.join(args.savePath, 'pyavi')
	args.pyframesPath 	= os.path.join(args.savePath, 'pyframes')
	args.pyworkPath 		= os.path.join(args.savePath, 'pywork')
	args.pycropPath 		= os.path.join(args.savePath, 'pycrop')
	args.videoFilePath 	= os.path.join(args.pyaviPath, 'video.avi')
	args.audioFilePath 	= os.path.join(args.pyaviPath, 'audio.wav')

	cam = cv2.VideoCapture(args.videoPath)
	args.fps = cam.get(cv2.CAP_PROP_FPS)


	if args.csvPath == None:
		args.csvPath = args.savePath + '/csv'

	return args


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


def main():
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

	# Initialization
	# global args
	args = initialize_args()

	if os.path.exists(args.savePath):
		rmtree(args.savePath)

	os.makedirs(args.pyaviPath, 		exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyframesPath, 	exist_ok = True) # Save all the video frames
	os.makedirs(args.pyworkPath, 		exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(args.pycropPath, 		exist_ok = True) # Save the detected face clips (audio+video) in this process

	# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	duration_args = '-ss {args.start} -to {args.start + args.duration}'
	if args.duration == 0:
		duration_args = ''

	# Extract video
	command = f"ffmpeg -y -i {args.videoPath} -qscale:v 2 -threads {args.nDataLoaderThread} {duration_args} -async 1 -r {args.fps} {args.videoFilePath} -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)
	report_save('video', args.videoFilePath)

	# Extract audio
	command = f"ffmpeg -y -i {args.videoFilePath} -qscale:a 0 -ac 1 -vn -threads {args.nDataLoaderThread} -ar 16000 {args.audioFilePath} -loglevel panic"
	subprocess.call(command, shell=True, stdout=None)
	report_save('audio', args.audioFilePath)

	# Extract the video frames
	command = f"ffmpeg -y -i {args.videoFilePath} -qscale:v 2 -threads {args.nDataLoaderThread} -f image2 {args.pyframesPath}/%06d.jpg -loglevel panic"
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

		if frame_count < args.minTrack:
			continue

		allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	log(f"Face track detected {len(allTracks)} tracks")

	# Face clips cropping
	vidTracks = []
	for index, track in tqdm(enumerate(allTracks), total = len(allTracks), desc='Clipping faces', leave=False, disable=not verbose):
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

	# Visualization, save the result as the new video
	if args.visualize:
		log('Rendering vizualization')
		visualization(vidTracks, scores, args)

	log('Rendering track csv')
	build_csv(vidTracks, scores, args)

if __name__ == '__main__':
		main()
