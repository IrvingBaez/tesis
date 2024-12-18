import torch
import psutil
import torch.multiprocessing as mp

# from model.avd.align_faces import main as align_faces
# from model.avd.extract_faces import main as extract_faces
# from model.asd.perform_asd import main as perform_asd
from model.avd.perform_avd import main as perform_avd
from model.third_party.avr_net.train import main as train_avd
# from model.asd.visualize_asd import main as visualize_asd
# from model.denoise.denoise import main as denoise
# from model.tools.der_and_losses import main as validation
# from model.util import get_path
# from model.third_party.pytorch_parallel.example import main as parallel_example

# TODO: Implement this process for unnanotated videos.
if __name__=='__main__':
	mp.set_start_method('spawn')

	data_type = 'val'

	print('\n\n0- SANITY CHECK')
	GB_FACTOR = 1024**3

	memory = psutil.virtual_memory()
	print(f"Total Memory:           {memory.total / GB_FACTOR:.2f} GB")
	print(f"Used Memory:            {memory.used / GB_FACTOR:.2f} GB")
	print(f"Available Memory:       {memory.available / GB_FACTOR:.2f} GB")
	print(f"Free Memory:            {memory.free / GB_FACTOR:.2f} GB")

	print(f'\nCUDA available:       {torch.cuda.is_available()}')
	print(f'Device count:           {torch.cuda.device_count()}')

	for i in range(torch.cuda.device_count()):
		print(f'\nInfo Device {i}:')
		print(f'    Device name:      {torch.cuda.get_device_name(i)}')
		print(f'    Available Memory: {torch.cuda.get_device_properties(i).total_memory / GB_FACTOR:.2f} GB')


	print('\n\n1- DENOISE WAVES')
	# for denoiser in ['noisereduce', 'dihard18']:
	# 	denoise(denoiser=denoiser)


	print('\n\n2- VOICE ACTIVITY DETECTION')
	print('Performed with dihard18')


	print('\n\n3- ACTIVE SPEAKER DETECTION')
	# for asd_detector in ['talk_net', 'light_asd']:
	# 	perform_asd(data_type=data_type, asd_detector=asd_detector, workers=1)


	print('\n\n4- FACE CROPPING')
	# for asd_detector in ['ground_truth', 'talk_net', 'light_asd']:
	# 	extract_faces(data_type=data_type, asd_detector=asd_detector)

	print('\n\n5- FACE ALIGN')
	# for asd_detector in ['ground_truth']:#, 'light_asd', 'talk_net']:
	# 	align_faces(asd_detector=asd_detector)

	print('\n\n6- TRAINING AUDIO VISUAL DIARIZATION')
	params = {
		# Data config
		'checkpoint': '',
		'disable_pb': True,
		# Architecture
		'self_attention': 'class_token',
		'cross_attention': '',
		# Hyperparams
		'learning_rate': 0.0005,
		'momentum': 0.05,
		'weight_decay': 0.0001,
		'step_size': 2,
		'gamma': 0.5,
		'epochs': 10,
		'frozen_epochs': 5,
		'video_proportion': 0.1,
		# 'frames': 5
		'aligned': False,
	}
	train_avd(**params)

	print('\n\n7- AUDIO VISUAL DIARIZATION')
	avd_tests = []
	for vad_detector in ['ground_truth']:#, 'dihard18']:
		for asd_detector in ['ground_truth']:#, 'light_asd', 'talk_net']:
			for denoiser in ['dihard18']:#, 'noisereduce', 'original']:
				for aligned in [True]: # False]:
					avd_tests.append({
						'data_type': 		data_type,
						'denoiser': 		denoiser,
						'vad_detector': vad_detector,
						'asd_detector': asd_detector,
						'aligned': 			aligned,
						'avd_detector': 'avr_net',
						'checkpoint':		'model/third_party/avr_net/checkpoints/2024_11_24 15:11:30/2024_11_25_04:21:40_epoch_00029.ckpt',
						'self_attention': 'class_token',
						'cross_attention': 'fusion'
					})

	for params in avd_tests:
		print(params)
		# perform_avd(**params)
		print('')


	# print('\n\n8- VISUALIZATION')
	# visualize_asd(
	# 	video_path='dataset/videos/1j20qq1JyX4_01.mp4',
	# 	csv_path='dataset/asd/ground_truth/predictions/1j20qq1JyX4_01.csv',
	# 	# gt_path=None,
	# 	output_path='1j20qq1JyX4_01.avi',
	# 	audio_path='dataset/waves/original/1j20qq1JyX4_01.wav')
