import torch
import psutil
import torch.multiprocessing as mp
import subprocess


# from model.avd.align_faces import main as align_faces
# from model.avd.extract_faces import main as extract_faces
# from model.asd.perform_asd import main as perform_asd
from model.avd.perform_avd import main as perform_avd
from model.third_party.avr_net.train import main as train_avd
from model.third_party.avr_net.train_lightning import main as train_lightning_avd
from model.third_party.avr_net.train_features_extraction import main as train_features_extraction
# from model.asd.visualize_asd import main as visualize_asd
# from model.denoise.denoise import main as denoise
# from model.tools.der_and_losses import main as validation
# from model.util import get_path


if __name__=='__main__':
	# mp.set_start_method('spawn', force=True)

	data_type = 'val'

	# result = subprocess.run(
	# 	['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
	# 	capture_output=True,
	# 	text=True
	# )
	# driver_version = result.stdout.strip()

	# result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
	# nvcc_version = result.stdout.strip()


	print('\n\n0- SANITY CHECK')
	GB_FACTOR = 1024**3

	memory = psutil.virtual_memory()
	print(f'Total Memory:           {memory.total / GB_FACTOR:.2f} GB')
	print(f'Used Memory:            {memory.used / GB_FACTOR:.2f} GB')
	print(f'Available Memory:       {memory.available / GB_FACTOR:.2f} GB')
	print(f'Free Memory:            {memory.free / GB_FACTOR:.2f} GB')

	print(f'\nCUDA available:       {torch.cuda.is_available()}')
	# print(f'\nCUDA version:      		{torch.version.cuda}')
	# print(f'\ntorch version:       	{torch.__version__}')
	# print(f'NVIDIA driver version:  {driver_version}')
	# print(f'NVCC version:						{nvcc_version}')
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


	print('\n\n6- Data Base Exploration')
	# train_features_extraction(disable_pb=True, aligned=False)


	# Self attention:
	# 	Funciones de pérdida dejar SGD.
	#		Probar Learning rate más bajo.
	# 	Usar promedio aritmético, pooling layer, convolution y self-attention.
	# 	Graficar promedio de entrenamiento y validación en una sola gráfica.
	print('\n\n6- TRAINING AUDIO VISUAL DIARIZATION')
	train_params = {
		# Data config
		'video_proportion': 			1.0,
		'val_video_proportion':		1.0,
		'aligned': 								False,
		'checkpoint': 						'',
		'max_frames': 						1,
		'disable_pb': 						False,
		# Architecture
		'self_attention': 				'pick_first', # 'class_token' or 'pick_first',
		'self_attention_dropout': 0.2,
		'cross_attention':	 			'concat', 		# 'fusion' or 'concat'
		# Hyperparams
		'loss_fn':								'mse', 				# 'bce' 'mse'
		'optimizer':							'sgd', 				# 'sgd' 'adam'
		'learning_rate': 					0.002,
		'momentum': 							0.005,				# Not working
		'weight_decay': 					0.0001,
		'step_size': 							4,
		'gamma': 									0.6,
		'epochs': 								15,
		'frozen_epochs': 					0,						# Not working
	}

	print('Starting training with params: ', train_params)
	# checkpoint = train_avd(**train_params)
	train_lightning_avd(**train_params)

	print('\n\n7- AUDIO VISUAL DIARIZATION')
	checkpoints = [
		'model/third_party/avr_net/checkpoints/lightning_logs/sgd+mse_loss/checkpoints/epoch=14-step=537810.ckpt'
	]
	avd_tests = []
	for vad_detector in ['ground_truth']:#, 'dihard18']:
		for asd_detector in ['ground_truth']:#, 'light_asd', 'talk_net']:
			for denoiser in ['dihard18']:#, 'noisereduce', 'original']:
				for aligned in [False]: # True]:
					for checkpoint in checkpoints:
						avd_tests.append({
							'data_type': 				'val',
							'denoiser': 				denoiser,
							'vad_detector': 		vad_detector,
							'asd_detector': 		asd_detector,
							'aligned': 					aligned,
							'checkpoint':				checkpoint
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
