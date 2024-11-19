import os
import torch
import psutil
from PIL import Image


components = {
	'data_type':			['test', 'val', 'train', 'demo'],
	'request':				['videos_path', 'waves_path', 'vad_path', 'asd_path', 'avd_path'],
	'denoiser': 			['original', 'dihard18', 'noisereduce'],
	'vad_detector':		['ground_truth', 'dihard18'],
	'asd_detector':		['ground_truth', 'light_asd', 'talk_net'],
	'avd_detector':		['ground_truth', 'avr_net', 'avar_net'],
}

def get_path(request, data_type='test', denoiser='original', vad_detector='ground_truth', asd_detector='ground_truth', avd_detector='avr_net'):
	if request 			not in components['request']: 			raise ValueError('Invalid value for request')
	if data_type 		not in components['data_type']: 		raise ValueError('Invalid value for data_type')
	if denoiser 		not in components['denoiser']: 			raise ValueError('Invalid value for denoiser')
	if vad_detector not in components['vad_detector']: 	raise ValueError('Invalid value for vad_detector')
	if asd_detector not in components['asd_detector']: 	raise ValueError('Invalid value for asd_detector')
	if avd_detector not in components['avd_detector']: 	raise ValueError('Invalid value for avd_detector')

	paths = {
		'videos_path':f'dataset/videos',
		'waves_path': f'dataset/waves/{denoiser}',
		'vad_path': 	f'dataset/vad/{vad_detector}',
		'asd_path': 	f'dataset/asd/{asd_detector}',
		'avd_path': 	f'dataset/avd/{avd_detector}',
	}

	path = paths[request]

	if data_type == 'demo':
		path = path.replace('dataset/', '')

	os.makedirs(path, exist_ok=True)

	return path


def intersection_over_union(boxA, boxB, relativeToBoxA=False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	if relativeToBoxA == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou


def argparse_helper(parser, **kwargs):
	if kwargs:
		args_list = []

		for key, value in kwargs.items():
			if key == 'not_empty': continue

			args_list.append(f'--{key}')

			if isinstance(value, bool):
				if not value:
					args_list.pop()
				continue

			if isinstance(value, str):
				if value.startswith('-'):
					args_list[-1] += f'={value}'
				else:
					args_list.append(value)

			if isinstance(value, int): args_list.append(str(value))
			if isinstance(value, float): args_list.append(str(value))
	else:
		args_list = None

	args = parser.parse_args(args=args_list)

	return args


def save_data(data, path):
	if os.path.exists(path):
		print(f"File {path} already exists. Skipping save to avoid overwrite.")
	else:
		torch.save(data, path)


def check_system_usage():
	print('\n\n================ CURRENT SYSYEM USAGE ================')
	GB_FACTOR = 1024**3
	system_total = 0.0
	system_reserved = 0.0
	system_available = 0.0

	memory = psutil.virtual_memory()
	print(f'\nSystem RAM Info:')
	print(f'Total Memory: {memory.total / GB_FACTOR:.2f} GB')
	print(f'Reserved Memory: {memory.used / GB_FACTOR:.2f} GB')
	print(f'Available Memory: {memory.available / GB_FACTOR:.2f} GB')

	info_lines = []
	for i in range(torch.cuda.device_count()):
		total_memory = torch.cuda.get_device_properties(i).total_memory / GB_FACTOR
		reserved_memory = torch.cuda.memory_reserved(i) / GB_FACTOR
		available_memory = total_memory - reserved_memory

		system_total += total_memory
		system_reserved += reserved_memory
		system_available += available_memory

		info_lines.append(f'\n  GPU {i}:')
		info_lines.append(f'    Device name:      {torch.cuda.get_device_name(i)}')
		info_lines.append(f'    Total Memory: {total_memory:.2f} GB')
		info_lines.append(f'    Reserved Memory: {reserved_memory:.2f} GB')
		info_lines.append(f'    Available Memory: {available_memory:.2f} GB')

	print(f'\nSystem VRAM Info:')
	print(f'Total Memory: {system_total:.2f} GB')
	print(f'Reserved Memory: {system_reserved:.2f} GB')
	print(f'Available Memory: {system_available:.2f} GB')

	for line in info_lines:
		print(line)


def show_similarities(folder, similarities):
	os.makedirs(folder, exist_ok=True)

	for key, value in similarities.items():
		tensor_img = (value * 255).byte()
		image = Image.fromarray(tensor_img.numpy(), mode='L')
		image.save(f"{folder}/{key}.png")