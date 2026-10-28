import json
import os

from tabulate import tabulate

from model.runs import RUNS, get_run
from model.third_party.avr_net.train_lightning import CHECKPOINT_DIR, best_val_summary, epoch_summaries, run_dir


def run_state(run_id):
	"""
	'new':			never started.
	'done':			finished all its epochs.
	'resumable':	stopped after at least one epoch, continues from last.ckpt.
	'broken':		stopped before finishing its first epoch, nothing to recover.
	"""
	folder = run_dir(run_id)

	if not os.path.isdir(folder):											return 'new'
	if os.path.exists(os.path.join(folder, 'done.json')):					return 'done'
	if os.path.exists(os.path.join(folder, 'checkpoints', 'last.ckpt')):	return 'resumable'

	return 'broken'


def _load_json(path):
	with open(path, 'r') as file:
		return json.load(file)


def _duration(seconds):
	if not seconds: return '-'
	minutes = int(seconds // 60)

	return f'{minutes // 60}h{minutes % 60:02d}m'


def _optimizer(config):
	text = f"{config['optimizer']} {config['learning_rate']:g}"
	if config['optimizer'] == 'sgd' and config['momentum']: text += f" m{config['momentum']:g}"

	return text


def _attention(config):
	parts = []
	if config['self_attention'] in ('class_token', 'temporal'): parts.append(config['self_attention'])
	if config['cross_attention'] == 'fusion': parts.append('fusion')

	return '+'.join(parts) if parts else '-'


def summary_rows():
	rows = []

	for run_id in RUNS:
		state = run_state(run_id)
		config_path = os.path.join(run_dir(run_id), 'config.json')
		config = _load_json(config_path) if os.path.exists(config_path) else get_run(run_id)
		epochs = epoch_summaries(run_dir(run_id))

		row = {
			'Run':			run_id,
			'Status':		{'new': 'pending', 'done': 'done', 'resumable': 'stopped', 'broken': 'stopped'}[state],
			'Loss':			config['loss_fn'],
			'Optim':		_optimizer(config),
			'Videos':		f"{config['video_proportion']:.0%}",
			'Frames':		config['db_video_mode'],
			'Attention':	_attention(config),
			'Epochs':		f"{len(epochs)}/{config['epochs']}",
		}

		if epochs:
			best = best_val_summary(run_dir(run_id))
			row.update({
				'Best ep':	best['epoch'],
				'AHC thr':	best['threshold'],
				'DER val':	best['der'],
				'Miss':		best['miss'],
				'FA':		best['fa'],
				'Conf':		best['confusion'],
				'F1 val':	best['f1'],
				'Time':		_duration(sum(summary.get('epoch_seconds', 0) for summary in epochs)),
			})

			# Train DER of the best epoch, computed once after the last epoch
			train_path = os.path.join(run_dir(run_id), 'train', 'summary.json')
			if os.path.exists(train_path):
				row['DER train'] = _load_json(train_path)['der']

			# Test DER with the threshold chosen in validation
			test_path = os.path.join(run_dir(run_id), 'test', 'summary.json')
			if os.path.exists(test_path):
				row['DER test'] = _load_json(test_path)['results'][f"{best['threshold']}"]['der']

		rows.append(row)

	return rows


def print_summary():
	columns = ['Run', 'Status', 'Loss', 'Optim', 'Videos', 'Frames', 'Attention', 'Epochs', 'Best ep', 'AHC thr',
		'DER train', 'DER val', 'Miss', 'FA', 'Conf', 'F1 val', 'DER test', 'Time']
	def cell(row, column):
		value = row.get(column, '-')
		if isinstance(value, float): return f'{value:.3f}' if column == 'AHC thr' else f'{value:.2f}'
		return value

	rows = [[cell(row, column) for column in columns] for row in summary_rows()]
	table = tabulate(rows, headers=columns, disable_numparse=True)

	text = (
		'\n\nEXPERIMENTS SUMMARY (all registered runs, read from disk)\n'
		'DER, Miss, FA and Conf in %, from the best validation epoch and its AHC threshold. '
		'DER train and DER test use that same epoch and threshold.\n\n'
		f'{table}\n'
	)
	print(text)

	os.makedirs(os.path.join(CHECKPOINT_DIR, 'lightning_logs'), exist_ok=True)
	with open(os.path.join(CHECKPOINT_DIR, 'lightning_logs', 'summary.txt'), 'w') as file:
		file.write(text)


if __name__ == '__main__':
	print_summary()
