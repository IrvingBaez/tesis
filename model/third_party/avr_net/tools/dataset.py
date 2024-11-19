from torch.utils.data.dataset import Dataset
from collections import defaultdict
from model.third_party.avr_net.tools.processor import *
import numpy as np
import random
import soundfile
import torch
import glob
import cv2
from tqdm.auto import tqdm


class CustomDataset(Dataset):
	def __init__(self, config, training=True, video_proportion=1.0, disable_pb=False):
		super().__init__()
		self.snippet_length	= 1
		self._max_frames		= 200
		self._min_frames		= 20
		self._step_frame		= 50

		# TODO: Implement
		self.missing_rate 	= 0
		self.max_utterance_frames = 1

		self.video_ids		= config['video_ids']
		self.waves_path		= config['waves_path']
		self.rttms_path		= config['rttms_path']
		self.labs_path		= config['labs_path']
		self.frames_path	= config['frames_path']
		self.disable_pb		= disable_pb

		self.training = training
		assert self.training == bool(self.rttms_path), 'Training mode requires rttms_path'

		assert 0.0 < video_proportion <= 1.0, 'Video proportion must be in the interval (0,1]'
		new_list_size = int(len(self.video_ids) * video_proportion)
		self.video_ids = random.sample(self.video_ids, new_list_size)

		self.processors = []
		self.processors.append(FacePad({'length': 1}))
		self.processors.append(FaceToTensor())
		self.processors.append(FaceResize({'dest_size': [112, 112]}))
		self.processors.append(FaceNormalize({'mean': 0.5, 'std': 0.5}))
		self.processors.append(AudioNormalize({'desired_rms': 0.1, 'eps': 0.0001}))
		self.processors.append(AudioToTensor())

		self.items = []
		self.entity_to_index = []

		self._load_dataset()


	def __len__(self):
		return len(self.items)


	def __getitem__(self, index):
		sample = self._load_video(index)

		for processor in self.processors:
			sample = processor(sample)

		sample['targets'] = self._get_target(index) if self.training else None
		sample['meta'].update(self._get_meta(index))

		return sample


	def _load_video(self, index):
		item = self.items[index]
		frames = self._load_frames(item['image_paths'])

		sample = {
			'frames':		frames,
			'audio':		self._load_audio(item),
			'visible':	torch.tensor([len(frames) > 0]),
			'meta':			{}
		}

		return sample


	def _load_audio(self, item):
		audio, sample_rate = soundfile.read(item['wave_path'])

		start, duration = item['start'] - item['offset'], item['duration']
		start, duration = round(start, 6), round(duration, 6)
		audio = audio[int(start * sample_rate) : int((start + duration) * sample_rate)]

		audiosize = audio.shape[0]
		if audiosize == 0:
			raise RuntimeError(f'Audio length is zero, check the file {item['wave_path']}. With item: {item}')

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
		frames = []
		for frame_path in frame_paths[:self.max_utterance_frames]:
			frames.append(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB))

		for _ in range(len(frame_paths), self.max_utterance_frames):
			frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

		frames = np.array(frames, dtype=np.uint8)

		return frames


	def _get_target(self, index):
		uid = self.items[index]['entity_id']
		target = torch.LongTensor([self.entity_to_index[uid]])

		return target


	def _get_meta(self, index):
		item = self.items[index]

		if len(item['image_paths']) > 0:
			track, id, _, _, _ = item['image_paths'][0].split('/')[-1].rsplit('.', 1)[0].split(':')
			trackid = track + ':' + id
		else:
			trackid = 'NA'

		meta = {
			'video':		item['video_id'],
			'start':		item['start'],
			'end':			item['start'] + item['duration'],
			'trackid':	trackid
		}
		return meta


	def _load_dataset(self):
		waves_path 	= self.waves_path

		maxs = self._max_frames / 100.0
		mins = self._min_frames / 100.0
		step = self._step_frame / 100.0

		for video_id in tqdm(sorted(self.video_ids), desc='Loading dataset', leave=False, disable=self.disable_pb):
			offset = 600.0 + int(video_id[-2:]) * 300.0

			if self.training:
				speech_segments, entity_ids = self._rttm_speech_segments(video_id)
			else:
				speech_segments = self._vad_speech_segments(video_id)

			video_faces = self._video_faces(video_id)

			for index, (start, end) in enumerate(speech_segments):
				# Splits segments if they are too long (2.0s), skips them if they are too short (0.2s)
				for seg_start in np.arange(start, end, step):
					seg_start = max(seg_start, offset)
					duration = round(min(maxs, end - seg_start), 6)

					if offset + 300.0 < seg_start + duration: continue
					if duration < mins: continue

					if self.training:
						self.entity_to_index.append(entity_ids[index])

					faces_in_segment = self._faces_in_segment(video_faces, seg_start, duration)

					if len(faces_in_segment) > 0:
						for _, image_paths in faces_in_segment.items():
							self.items.append({
								'wave_path':		f'{waves_path}/{video_id}.wav',
								'start':				seg_start,
								'duration':			duration,
								'offset':				offset,
								'image_paths':	image_paths,
								'video_id':			video_id,
								'entity_id':		entity_ids[index] if self.training else None
							})
					else:
						# offscreen speaker
						self.items.append({
							'wave_path':		f'{waves_path}/{video_id}.wav',
							'start':				seg_start,
							'duration':			duration,
							'offset':				offset,
							'image_paths':	[],
							'video_id':			video_id,
							'entity_id':		entity_ids[index] if self.training else None
						})

		if self.training:
			self.entity_to_index = sorted(list(set(self.entity_to_index)))
			self.entity_to_index = { uid: i for i, uid in enumerate(self.entity_to_index) }


	def _vad_speech_segments(self, video_id):
		vad_path = f'{self.labs_path}/{video_id}.lab'

		with open(vad_path, 'r') as f:
			lines = f.readlines()

		segments = []
		for line in lines:
			start, end, _ = line.split()
			segments.append((float(start), float(end)))

		return segments


	def _rttm_speech_segments(self, video_id):
		rttm_path = f'{self.rttms_path}/{video_id}.rttm'

		with open(rttm_path, 'r') as f:
			lines = f.readlines()

		segments = []
		entity_ids = []
		for line in lines:
			_, _, _, start, duration, _, _, spk_id, _, _ = line.split()
			segments.append((float(start), float(start) + float(duration)))
			entity_ids.append(f'{video_id}:{spk_id}')

		return segments, entity_ids


	def _video_faces(self, video_id):
		image_paths = glob.glob(f'{self.frames_path}/{video_id}/*.*')
		image_paths = sorted(image_paths)
		faces = defaultdict(lambda: defaultdict(list))

		for image_path in image_paths:
			# ['2XeFK-DTSZk_0960_1020', '23', '983.89', '1', 'spk01']
			track, id, timestamp, _, _ = image_path.split('/')[-1].rsplit('.', 1)[0].split(':')
			trackid = f'{track}:{id}'
			faces[float(timestamp)][trackid].append(image_path)

		timestamps = np.sort(np.array(list(faces.keys())))

		return faces, timestamps


	def _faces_in_segment(self, video_faces, start, duration):
		faces, timestamps = video_faces

		segment_timestamps = timestamps[np.searchsorted(timestamps, start) : np.searchsorted(timestamps, round(start + duration, 6))]
		segment_faces = defaultdict(list)

		for timestamp in segment_timestamps:
			face = faces[timestamp]
			for key, value in face.items():
				# list of dicts, where keys are trackid and values are image paths.
				segment_faces[key].extend(value)

		return segment_faces
