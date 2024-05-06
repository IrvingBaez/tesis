from torch.utils.data.dataset import Dataset
from collections import defaultdict
from model.third_party.avr_net.tools.processor import *
import numpy as np
import soundfile
import torch
import glob
import cv2

# Documentation: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files


class CustomDataset(Dataset):
	# DATASET CONFIG: {'data_dir': './dataset/', 'max_frame': 200, 'min_frame': 20, 'step_frame': 50, 'snippet_length': 1, 'missing_rate': 0, 'processors': {'face_pad_processor': {'type': 'face_pad', 'params': {'length': 1}}, 'face_to_tensor_processor': {'type': 'face_to_tensor', 'params': {}}, 'face_resize_processor': {'type': 'face_resize', 'params': {'dest_size': [112, 112]}}, 'face_normalize_processor': {'type': 'face_normalize', 'params': {'mean': 0.5, 'std': 0.5}}, 'audio_normalize_processor': {'type': 'audio_normalize', 'params': {'desired_rms': 0.1, 'eps': 0.0001}}, 'audio_to_tensor_processor': {'type': 'audio_to_tensor', 'params': {}}}, 'sampler': {}}
	def __init__(self, data_path, detector='ground_truth'):
		super().__init__()
		self.batch_size = 10
		self.snippet_length = 1
		self.data_path = data_path
		self.detector = detector
		self._max_frames = 200
		self._min_frames = 20
		self._step_frame = 50
		self._missing_rate = 0

		self.processors = []
		self.processors.append(FacePad({'length': 1}))
		self.processors.append(FaceToTensor())
		self.processors.append(FaceResize({'dest_size': [112, 112]}))
		self.processors.append(FaceNormalize({'mean': 0.5, 'std': 0.5}))
		self.processors.append(AudioNormalize({'desired_rms': 0.1, 'eps': 0.0001}))
		self.processors.append(AudioTransform())


	def __len__(self):
		return len(self.items)


	def __getitem__(self, indices):
		if isinstance(indices, int): indices = [[indices]]

		samples = [[self._get_one_item(index) for index in pair] for pair in indices]

		return samples


	def _get_one_item(self, index) -> dict:
		sample = self._load_video(self.items[index])

		for processor in self.processors:
			sample = processor(sample)

		sample['targets'] = None
		sample['meta'].update(self._get_meta(index))

		return sample


	def _load_video(self, item: tuple) -> dict:
		"""Loads one frame-audio pair"""

		audio = self._load_wave(item)
		frames = self._load_frames(item[4])
		sample = {}

		sample['frames'] = frames
		sample['audio'] = audio
		sample['meta'] = { 'visible': len(frames) > 0 }

		return sample


	def _load_wave(self, item):
		audio, sample_rate = soundfile.read(item[0])

		start, dura = item[1] - item[3], item[2]
		audio = audio[int(start * sample_rate) : int((start + dura) * sample_rate)]

		audiosize = audio.shape[0]
		if audiosize == 0: raise RuntimeError('Audio length is zero, check the file')

		max_audio = self._max_frames * int(sample_rate / 100)

		if audiosize < max_audio:
			shortage = max_audio - audiosize
			audio = np.pad(audio, (0, shortage), 'wrap')
			audiosize = audio.shape[0]

		if audiosize >= max_audio:
			startframe = int(torch.rand([1]).item()*(audiosize-max_audio))
			audio = audio[startframe: startframe+max_audio]

		return audio


	def _load_frames(self, frames):
		if len(frames) > 0:
			indices = torch.randint(0, len(frames), [self.snippet_length])
			frames = np.array([cv2.cvtColor(cv2.imread(frames[i]), cv2.COLOR_BGR2RGB) for i in indices])
			frames = center_pad(frames)
		else:
			# TODO: The zero-filling should occur here
			frames = np.array([], dtype=np.uint8)

		return frames


	def _get_meta(self, index):
		if len(self.items[index][-2]) > 0:
			track, id, _, _, _ = self.items[index][-2][0].split('/')[-1].rsplit('.', 1)[0].split(':')
			trackid = track + ':' + id
		else:
				trackid = 'NA'
		meta = {
			'video': self.items[index][-1],
			'start': self.items[index][1],
			'end': self.items[index][1] + self.items[index][2],
			'trackid': trackid
		}
		return meta


	# TODO: this function does too much stuff.
	def load_dataset(self):
		videos	= glob.glob(f'{self.data_path}/videos/*.*')
		waves_path = f'{self.data_path}/waves'
		vad_path = f'{self.data_path}/labs'

		maxs = self._max_frames / 100.0
		mins = self._min_frames / 100.0
		step = self._step_frame / 100.0

		self.items = []

		# parse audio and face segments
		for video in videos:
			video_name = video.split('/')[-1].split('.')[0]

			vad_file_path = f'{vad_path}/{video_name}.lab'

			with open(vad_file_path, 'r') as f:
				speech_segments = f.readlines()

			offset = float(speech_segments[0].split()[0])

			image_paths = glob.glob(f'{self.data_path}/asd/{self.detector}/aligned_tracklets/{video_name}/*')
			faces = defaultdict(lambda: defaultdict(list))

			for image_path in image_paths:
				# ['2XeFK-DTSZk_0960_1020', '23', '983.89', '1', '01spk01']
				track, id, timestamp, _, _ = image_path.split('/')[-1].rsplit('.', 1)[0].split(':')
				trackid = f'{track}:{id}'
				faces[float(timestamp)][trackid].append(image_path)

			for segment in speech_segments:
				item = segment.split()
				start = float(item[0])
				end = float(item[1])

				items = []
				timestamps = np.sort(np.array(list(faces.keys())))

				for seg_start in np.arange(start, end, step):
					duration = min(maxs, end - seg_start)
					if duration < mins: continue

					segment_timestamps = timestamps[np.searchsorted(timestamps, seg_start) : np.searchsorted(timestamps, seg_start+duration)]
					segment_faces = defaultdict(list)

					for timestamp in segment_timestamps:
						face = faces[timestamp]
						for key, value in face.items():
							# list of dicts, where keys are trackid and values are image paths.
							segment_faces[key].extend(value)

					if len(segment_faces) > 0:
						for _, image_paths in segment_faces.items():
							items.append((f'{waves_path}/{video_name}.wav', seg_start, duration, offset, image_paths, video_name))

				# Tuple: (wave_path, segment_start, segment_duration, offset, [image paths], video_id)
				self.items.extend(items)


# TODO: Processor should get rid of this.
def center_pad(frame, target_height=224, target_width=224):
    """
    Pads the middle two dimensions of a 4-dimensional array to the target dimensions.
    :param frame: 4-dimensional numpy array of shape (N, H, W, C)
    :param target_height: desired height after padding
    :param target_width: desired width after padding
    :return: padded array with shape (N, target_height, target_width, C)
    """
    # Extract the original dimensions
    n, h, w, c = frame.shape

    # Calculate padding sizes for height
    pad_h1 = (target_height - h) // 2
    pad_h2 = target_height - pad_h1 - h

    # Calculate padding sizes for width
    pad_w1 = (target_width - w) // 2
    pad_w2 = target_width - pad_w1 - w

    # Apply padding to the second (h) and third (w) dimensions
    padded_frame = np.pad(frame, pad_width=((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode='constant', constant_values=0)

    return padded_frame
