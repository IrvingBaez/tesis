import argparse, os, subprocess, shutil
from tqdm.contrib.concurrent import process_map
from pathlib import Path
from glob import glob
import model.util as util
from scipy.io import wavfile
import noisereduce


def denoise(args):
	extract_waves(args)

	if args.denoiser == 'dihard18':
		dihard18_denoise(args)
	elif args.denoiser == 'noisereduce':
		noisereduce_denoise(args)
	elif args.denoiser == 'original':
		return


def extract_waves(args):
	extract_tasks = []

	for video_path in glob(f'{args.videos_path}/*.*'):
		uid = os.path.basename(video_path).split('.')[0]
		original_wave_path = f'{args.original_waves_path}/{uid}.wav'

		if not os.path.exists(original_wave_path):
			extract_tasks.append((video_path, original_wave_path))

	process_map(ffmpeg_command, extract_tasks, max_workers=args.n_threads, chunksize=1, desc='Extracting waves')


def ffmpeg_command(data):
		video, output_path = data
		subprocess.call(f'ffmpeg -y -i {video} -qscale:a 0 -ac 1 -vn -threads 1 -ar 16000 {output_path} -loglevel panic'.split())


def dihard18_denoise(args):
	project_folder = str(Path(__file__).parents[2])

	original_path = project_folder + '/' + args.original_waves_path

	subprocess.call(f'docker build -t dihard18 -f model/third_party/dihard18/Dockerfile model/third_party/dihard18'.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	subprocess.call(f'docker run -it --rm --runtime=nvidia -v {original_path}:/data dihard18 /bin/bash -c ./run_eval.sh'.split())

	if os.path.exists(f'{args.vad_path}/predictions'):
		shutil.rmtree(f'{args.vad_path}/predictions')

	shutil.rmtree(args.denoised_waves_path)
	os.makedirs(f'{args.vad_path}/predictions')
	os.rename(f'{args.original_waves_path}/dihard18', args.denoised_waves_path)
	os.rename(f'{args.original_waves_path}/vad', f'{args.vad_path}/predictions')


def noisereduce_denoise(args):
	denoise_tasks = []
	for original_path in glob(f'{args.original_waves_path}/*.*'):
		denoised_path = args.denoised_waves_path + '/' + os.path.basename(original_path)

		if not os.path.exists(denoised_path):
			denoise_tasks.append((original_path, denoised_path))

	process_map(ffmpeg_command, denoise_tasks, max_workers=args.n_threads, chunksize=1, desc='Denoising with noisereduce')


def apply_noisereduce(data):
	original_path, denoised_path = data

	rate, file = wavfile.read(original_path)

	wave = noisereduce.reduce_noise(y=file, sr=rate)
	wavfile.write(denoised_path, rate, wave)


def initialize_arguments(**kwargs):
	parser = argparse.ArgumentParser(description = "Arguments for data preprocessing")

	parser.add_argument('--data_type',	type=str, default="val",			help='Location of dataset to process')
	parser.add_argument('--denoiser',		type=str, default="original",	help='Location of dataset to process')
	parser.add_argument('--n_threads',	type=int, default=6,					help='Number of threads for preprocessing')

	args = util.argparse_helper(parser, **kwargs)

	args.videos_path = util.get_path('videos_path', data_type=args.data_type)
	args.original_waves_path = util.get_path('waves_path', data_type=args.data_type, denoiser='original')
	args.denoised_waves_path = util.get_path('waves_path', data_type=args.data_type, denoiser=args.denoiser)

	if args.denoiser == 'dihard18':
		args.vad_path = util.get_path('vad_path', data_type=args.data_type, denoiser='dihard18', vad_detector='dihard18')

	return args


def main(**kwargs):
	args = initialize_arguments(**kwargs, not_empty=True)
	denoise(args)


if __name__ == '__main__':
	args = initialize_arguments()
	denoise(args)
