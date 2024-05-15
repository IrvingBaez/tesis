from model.denoise.denoise import main as denoise
from model.asd.perform_asd import main as perform_asd
from model.avd.extract_faces import main as extract_faces
from model.avd.perform_avd import main as perform_avd
from model.util import get_path

# TODO: Implement this process for unnanotated videos.
if __name__=='__main__':
	data_type = 'test'

	print('\n\n1- DENOISE WAVES')
	# for denoiser in ['noisereduce', 'dihard18']:
	# 	denoise(data_type=data_type, denoiser=denoiser)

	print('\n\n2- VOICE ACTIVITY DETECTION')
	print('Performed with dihard18')


	print('\n\n3- ACTIVE SPEAKER DETECTION')
	# for asd_detector in ['light_asd']:
	# 	perform_asd(data_type=data_type, asd_detector=asd_detector) #, visualize=True)


	print('\n\n4- FACE CROPPING')
	for asd_detector in ['ground_truth', 'light_asd']:
		extract_faces(data_type=data_type, asd_detector=asd_detector)

	print('\n\n5- AUDIO VISUAL DIARIZATION')
	avd_tests = [
		{'data_type': data_type, 'denoiser': 'dihard18', 'vad_detector': 'ground_truth', 'asd_detector': 'ground_truth',	'avd_detector': 'avr_net'},
		{'data_type': data_type, 'denoiser': 'dihard18', 'vad_detector': 'ground_truth', 'asd_detector': 'light_asd', 		'avd_detector': 'avr_net'}
	]
	for params in avd_tests:
		perform_avd(**params)
		print('')


	print('\n\n6- SCORING')
	for params in avd_tests:
		scores_path = get_path('avd_path', **params) + '/scores.out'

		with open(scores_path, 'r') as file:
			lines = file.readlines()

		print('params:\t', params)
		print(lines[0].strip())
		print(lines[-1] + '\n')

