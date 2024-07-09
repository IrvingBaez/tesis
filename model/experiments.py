from model.denoise.denoise import main as denoise
from model.asd.perform_asd import main as perform_asd
from model.avd.extract_faces import main as extract_faces
from model.avd.align_faces import main as align_faces
from model.avd.perform_avd import main as perform_avd
from model.avd.train_avd_predictor import main as train_avd
from model.util import get_path
from model.asd.visualize_asd import main as visualize_asd


# TODO: Implement this process for unnanotated videos.
if __name__=='__main__':
	data_type = 'val'

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
	# for asd_detector in ['ground_truth', 'light_asd', 'talk_net']:
	# 	align_faces(asd_detector=asd_detector)

	print('\n\n6- TRAINING AUDIO VISUAL DIARIZATION')
	# params = {'denoiser': 'dihard18', 'vad_detector': 'ground_truth', 'asd_detector': 'ground_truth', 'avd_detector': 'avr_net', 'aligned': True}

	# train_avd(**params)


	print('\n\n6- AUDIO VISUAL DIARIZATION')
	avd_tests = []
	for vad_detector in ['ground_truth', 'dihard18']:
		for asd_detector in ['ground_truth', 'light_asd', 'talk_net']:
			for denoiser in ['dihard18', 'noisereduce', 'original']:
				for aligned in [False, True]:
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


	# print('\n\n7- VISUALIZATION')
	# visualize_asd(
	# 	video_path='dataset/videos/1j20qq1JyX4_01.mp4',
	# 	csv_path='dataset/asd/ground_truth/predictions/1j20qq1JyX4_01.csv',
	# 	# gt_path=None,
	# 	output_path='1j20qq1JyX4_01.avi',
	# 	audio_path='dataset/waves/original/1j20qq1JyX4_01.wav')
