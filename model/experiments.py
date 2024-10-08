import torch
import psutil

from model.avd.align_faces import main as align_faces
from model.avd.extract_faces import main as extract_faces
from model.asd.perform_asd import main as perform_asd
from model.avd.perform_avd import main as perform_avd
from model.avd.train_avd_predictor import main as train_avd
from model.asd.visualize_asd import main as visualize_asd
from model.denoise.denoise import main as denoise
from model.tools.der_and_losses import main as validation
from model.util import get_path
from model.third_party.pytorch_parallel.example import main as parallel_example

# TODO: Implement this process for unnanotated videos.
if __name__=='__main__':
	data_type = 'val'

	print('\n\n0- SANITY CHECK')
	GB_FACTOR = 1024**3

	memory = psutil.virtual_memory()
	print(f"Total Memory:           {memory.total / GB_FACTOR:.2f} GB")
	print(f"Used Memory:            {memory.used / GB_FACTOR:.2f} GB")
	print(f"Available Memory:       {memory.available / GB_FACTOR:.2f} GB")
	print(f"Free Memory:            {memory.free / GB_FACTOR:.2f} GB")

	print(f'\nCUDA available:         {torch.cuda.is_available()}')
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
	params = {'denoiser': 'dihard18', 'vad_detector': 'ground_truth', 'asd_detector': 'ground_truth', 'avd_detector': 'avr_net', 'aligned': True, 'epochs': 20, 'checkpoint': 'model/third_party/avr_net/weights/best_0.14_20.66.ckpt'}

	train_avd(**params)
	# parallel_example(checkpoint='model/third_party/pytorch_parallel/checkpoints/checkpoint_epoch_00004.pth')

	# print('\n\n8- VALIDATING AUDIO VISUAL DIARIZATION')
	# validation()

	print('\n\n9- AUDIO VISUAL DIARIZATION')
	avd_tests = []
	for vad_detector in ['ground_truth']:#, 'dihard18']:
		for asd_detector in ['ground_truth']:#, 'light_asd', 'talk_net']:
			for denoiser in ['dihard18']:#, 'noisereduce', 'original']:
				for aligned in [False]: # False]:
					avd_tests.append({
						'data_type': 		data_type,
						'denoiser': 		denoiser,
						'vad_detector': vad_detector,
						'asd_detector': asd_detector,
						'aligned': 			aligned,
						'avd_detector': 'avr_net'
					})

	for params in avd_tests:
		print(params)
		perform_avd(**params)
		print('')


	# print('\n\n8- VISUALIZATION')
	# visualize_asd(
	# 	video_path='dataset/videos/1j20qq1JyX4_01.mp4',
	# 	csv_path='dataset/asd/ground_truth/predictions/1j20qq1JyX4_01.csv',
	# 	# gt_path=None,
	# 	output_path='1j20qq1JyX4_01.avi',
	# 	audio_path='dataset/waves/original/1j20qq1JyX4_01.wav')
