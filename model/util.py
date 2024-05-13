import os

components = {
	'data_type':			['test', 'val', 'train', 'demo'],
	'request':				['videos_path', 'waves_path', 'vad_path', 'asd_path', 'avd_path'],
	'denoiser': 			['original', 'dihard18', 'noisereduce'],
	'vad_detector':		['ground_truth', 'dihard18'],
	'asd_detector':		['ground_truth', 'light_asd', 'talk_net'],
	'avd_detector':		['ground_truth', 'avr_net'],
}

def get_path(request, data_type='val', denoiser='dihard18', vad_detector='ground_truth', asd_detector='ground_truth', avd_detector='avr_net'):
	if request 			not in components['request']: 			raise ValueError('Invalid value for request')
	if data_type 		not in components['data_type']: 		raise ValueError('Invalid value for data_type')
	if denoiser 		not in components['denoiser']: 			raise ValueError('Invalid value for denoiser')
	if vad_detector not in components['vad_detector']: 	raise ValueError('Invalid value for vad_detector')
	if asd_detector not in components['asd_detector']: 	raise ValueError('Invalid value for asd_detector')
	if avd_detector not in components['avd_detector']: 	raise ValueError('Invalid value for avd_detector')

	paths = {
		'videos_path':f'dataset/{data_type}/videos',
		'waves_path': f'dataset/{data_type}/waves/{denoiser}',
		'vad_path': 	f'dataset/{data_type}/vad/{denoiser}/{vad_detector}',
		'asd_path': 	f'dataset/{data_type}/asd/{asd_detector}',
		'avd_path': 	f'dataset/{data_type}/avd/{denoiser}/{vad_detector}/{asd_detector}/{avd_detector}',
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
			if isinstance(value, str): args_list.append(value)
	else:
		args_list = None

	args = parser.parse_args(args=args_list)

	return args