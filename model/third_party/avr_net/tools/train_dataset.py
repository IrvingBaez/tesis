from torch.utils.data.dataset import Dataset
from collections import defaultdict
from model.third_party.avr_net.tools.processor import *
import numpy as np
import soundfile
import torch
import glob
import cv2

# Documentation: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files


class TrainDataset(Dataset):
	# DATASET CONFIG: {'data_dir': './dataset/', 'max_frame': 200, 'min_frame': 20, 'step_frame': 50, 'snippet_length': 1, 'missing_rate': 0, 'processors': {'face_pad_processor': {'type': 'face_pad', 'params': {'length': 1}}, 'face_to_tensor_processor': {'type': 'face_to_tensor', 'params': {}}, 'face_resize_processor': {'type': 'face_resize', 'params': {'dest_size': [112, 112]}}, 'face_normalize_processor': {'type': 'face_normalize', 'params': {'mean': 0.5, 'std': 0.5}}, 'audio_normalize_processor': {'type': 'audio_normalize', 'params': {'desired_rms': 0.1, 'eps': 0.0001}}, 'audio_to_tensor_processor': {'type': 'audio_to_tensor', 'params': {}}}, 'sampler': {}}
	def __init__(self, args):
		super().__init__()
		self.batch_size = 10
		self.snippet_length = 1
		self._max_frames = 200
		self._min_frames = 20
		self._step_frame = 50
		self._missing_rate = 0

		self.video_ids		= args.video_ids
		self.videos_path	= args.videos_path
		self.waves_path		= args.waves_path
		self.rttms_path		= args.rttms_path
		self.frames_path	= args.frames_path

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

		sample['targets'] = self._get_target(index)
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
		start, dura = round(start, 6), round(dura, 6)
		audio = audio[int(start * sample_rate) : int((start + dura) * sample_rate)]

		audiosize = audio.shape[0]
		if audiosize == 0:
			raise RuntimeError('Audio length is zero, check the file')

		max_audio = self._max_frames * int(sample_rate / 100)

		if audiosize < max_audio:
			shortage = max_audio - audiosize
			audio = np.pad(audio, (0, shortage), 'wrap')
			audiosize = audio.shape[0]

		if audiosize > max_audio:
			startframe = int(torch.rand([1]).item()*(audiosize-max_audio))
			audio = audio[startframe: startframe+max_audio]

		return audio


	def _load_frames(self, frame_paths):
		if len(frame_paths) > 0:
			indices = [0] # torch.randint(0, len(frame_paths), [self.snippet_length])
			frames = np.array([cv2.cvtColor(cv2.imread(frame_paths[i]), cv2.COLOR_BGR2RGB) for i in indices])
		else:
			frames = np.array([], dtype=np.uint8)

		return frames


	def _get_target(self, index):
		uid = self.items[index][-1]
		target =  torch.LongTensor([self.entity_to_index[uid]])

		return target


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
		waves_path 	= self.waves_path
		rttms_path 	= self.rttms_path

		maxs = self._max_frames / 100.0
		mins = self._min_frames / 100.0
		step = self._step_frame / 100.0

		self.items = []
		self.entity_to_index = []

		# parse audio and face segments
		for video_id in sorted(self.video_ids):
			rttm_file_path = f'{rttms_path}/{video_id}.rttm'

			with open(rttm_file_path, 'r') as f:
				utterances = f.readlines()

			# offset = float(speech_segments[0].split()[0])
			offset = 600.0 + float(video_id.rsplit('_', 1)[-1]) * 300.0

			image_paths = glob.glob(f'{self.frames_path}/{video_id}/*.*')
			image_paths = sorted(image_paths)
			faces = defaultdict(lambda: defaultdict(list))

			for image_path in image_paths:
				# ['2XeFK-DTSZk_0960_1020', '23', '983.89', '1', 'spk01']
				_, _, timestamp, _, spk_id = image_path.split('/')[-1].rsplit('.', 1)[0].split(':')
				faces[spk_id][float(timestamp)] = image_path

			for utterance in utterances:
				_, _, _, start, duration, _, _, spk_id, _, _ = utterance.split()

				start = float(start)
				end = start + float(duration)
				entity_id = f'{video_id}:{spk_id}'

				items = []
				timestamps = np.sort(np.array(list(faces[spk_id].keys())))

				# Splits segments if they are too long (0.5s), skips them if they are too short (0.2s)
				for seg_start in np.arange(start, end, step):
					if seg_start < offset: seg_start = offset
					duration = round(min(maxs, end - seg_start), 6)

					if start + duration > offset + 300.0: continue

					segment_timestamps = timestamps[np.searchsorted(timestamps, seg_start) : np.searchsorted(timestamps, round(seg_start+duration, 6))]
					image_paths = [faces[spk_id][timestamp] for timestamp in segment_timestamps]

					items.append((f'{waves_path}/{video_id}.wav', seg_start, duration, offset, image_paths, entity_id))
					self.entity_to_index.append(entity_id)

				# Tuple: (wave_path, segment_start, segment_duration, offset, [image paths], video_id)
				self.items.extend(items)

		self.entity_to_index = sorted(list(set(self.entity_to_index)))
		self.entity_to_index = { uid: i for i, uid in enumerate(self.entity_to_index) }

		self.video_to_entities = defaultdict(list)
		self.entity_to_indices = defaultdict(list)

		for i, item in enumerate(self.items):
			self.entity_to_indices[item[-1]].append(i)

		for entity_id in self.entity_to_indices.keys():
			video_id = entity_id.split(':')[0]
			self.video_to_entities[video_id].append(entity_id)
