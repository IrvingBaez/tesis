import argparse
import shutil
import traceback
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
from model.runs import RUNS, get_run
from model.run_summary import print_summary, run_state
from model.third_party.avr_net.train_lightning import run_dir
# from model.asd.visualize_asd import main as visualize_asd
# from model.denoise.denoise import main as denoise
# from model.util import get_path


if __name__=='__main__':
	# mp.set_start_method('spawn', force=True)

	parser = argparse.ArgumentParser(
		description='Runs the experiments registered in model/runs.py, one after another. Finished runs are skipped '
			'and stopped runs continue from their last checkpoint. Without run IDs, only prints the summary table.')
	parser.add_argument('runs', nargs='*', help=f'Run IDs to execute in order. Available: {", ".join(RUNS)}')
	cli = parser.parse_args()

	for run_id in cli.runs: get_run(run_id)  # Fail early on unknown IDs

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

	print('\n\n6- TRAINING AUDIO VISUAL DIARIZATION')
	failed = []

	try:
		for run_id in cli.runs:
			state = run_state(run_id)
			print(f'\n\n======================== RUN {run_id} ({state}) ========================')

			if state == 'done':
				print(f'{run_id} already finished, skipping it.')
				continue

			if state == 'broken':
				print(f'{run_id} stopped before finishing its first epoch, starting it again.')
				shutil.rmtree(run_dir(run_id))

			try:
				if state == 'resumable':
					trainer = train_lightning_avd(run_name=run_id, resume=True)
				else:
					trainer = train_lightning_avd(**get_run(run_id))
			except Exception:
				traceback.print_exc()
				failed.append(run_id)
				continue

			# Lightning catches Ctrl+C and returns, stop the queue instead of starting the next run
			if trainer.interrupted:
				print(f'\n{run_id} interrupted. Run the same command again to continue from its last checkpoint.')
				break
	finally:
		print_summary()

	if failed:
		raise SystemExit(f'Failed runs: {", ".join(failed)}')

	print('\n\n7- AUDIO VISUAL DIARIZATION')
	checkpoints = [
		'model/third_party/avr_net/checkpoints/lightning_logs/Reference/checkpoints/epoch=0-step=19637.ckpt'
	]

	# for checkpoint in checkpoints:
	# 	print(checkpoint)
	# 	perform_avd(aligned=False, checkpoint=checkpoint, max_frames=1, db_video_mode='keep_all')
	# 	print('')


	# print('\n\n8- VISUALIZATION')
	# visualize_asd(
	# 	video_path='dataset/videos/1j20qq1JyX4_01.mp4',
	# 	csv_path='dataset/asd/ground_truth/predictions/1j20qq1JyX4_01.csv',
	# 	# gt_path=None,
	# 	output_path='1j20qq1JyX4_01.avi',
	# 	audio_path='dataset/waves/original/1j20qq1JyX4_01.wav')
