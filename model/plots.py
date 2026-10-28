# Figures for the results chapter, read from the files that each run leaves in lightning_logs/<RUN_ID>/.
#
#   python model/plots.py histograms E2 E3 E5
#   python model/plots.py thresholds E2 E3 E4 E5 E6 E7
#   python model/plots.py curves E4
#
# A run can be given by ID or by folder (e.g. an archived run). Figures are saved as PDF in --out.

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from model.third_party.avr_net.train_lightning import best_val_summary, epoch_summaries, run_dir

DEFAULT_OUT = '../ThesisDoc/images/results'
FEATURES_DIR = 'model/third_party/avr_net/features'

# Categorical palette, light mode, fixed order (validated with the dataviz skill's validate_palette.js).
# Three slots sit below 3:1 contrast, so every series also gets its own marker and a legend entry.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style():
	plt.rcParams.update({
		'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': TEXT, 'axes.titlesize': 9,
		'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
		'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.5,
	})


def _folder(run):
	return run if os.path.isdir(run) else run_dir(run)


def _name(run):
	return os.path.basename(os.path.normpath(run))


def _config(run):
	with open(os.path.join(_folder(run), 'config.json'), 'r') as file:
		return json.load(file)


def _save(figure, out, filename):
	os.makedirs(out, exist_ok=True)
	path = os.path.join(out, filename)
	figure.savefig(path, bbox_inches='tight')
	plt.close(figure)
	print(f'Saved {path}')


def _pair_scores(run):
	"""Scores and labels (1 = same speaker) of every validation pair of the best epoch."""
	best = best_val_summary(_folder(run))
	data = torch.load(os.path.join(_folder(run), f"val_{best['epoch']}", 'similarities.pth'))

	config = _config(run)
	features = f"{FEATURES_DIR}/{config['max_frames']}_frames{'_aligned' if config['aligned'] else ''}_{config['db_video_mode']}/val"

	scores, labels = [], []
	for video_id, matrix in data['similarities'].items():
		targets = torch.load(f'{features}/{video_id}.pckl')['targets'].flatten()
		upper = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
		scores.append(matrix[upper])
		labels.append((targets[:, None] == targets[None, :])[upper])

	return torch.cat(scores), torch.cat(labels), best


def histograms(runs, out):
	"""One panel per run: similarity scores of positive and negative validation pairs, best epoch."""
	figure, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 2.6), sharey=True, squeeze=False)

	for axis, run in zip(axes[0], runs):
		scores, labels, best = _pair_scores(run)
		bins = torch.linspace(0, 1, 51).tolist()

		for (name, mask), color in zip([('Same speaker', labels), ('Different speaker', ~labels)], SERIES):
			axis.hist(scores[mask].numpy(), bins=bins, density=True, histtype='step', color=color, linewidth=1.5, label=name)

		axis.axvline(best['threshold'], color=MUTED, linestyle='--', linewidth=1)
		axis.set_title(f"{_name(run)} ({_config(run)['loss_fn']}), epoch {best['epoch']}", color=TEXT)
		axis.set_xlabel('Similarity score')
		axis.set_xlim(0, 1)

	axes[0][0].set_ylabel('Density')
	handles = [Line2D([], [], color=color, linewidth=1.5) for color in SERIES[:2]]
	handles.append(Line2D([], [], color=MUTED, linestyle='--', linewidth=1))
	figure.legend(handles, ['Same speaker', 'Different speaker', 'Chosen AHC threshold'], loc='upper center',
		bbox_to_anchor=(0.5, -0.06), ncol=3)
	_save(figure, out, f"histograms_{'_'.join(_name(run) for run in runs)}.pdf")


def thresholds(runs, out):
	"""Validation DER as a function of the AHC threshold, best epoch of each run."""
	figure, axis = plt.subplots(figsize=(4.5, 3))

	for index, run in enumerate(runs):
		best = best_val_summary(_folder(run))
		values = sorted((float(threshold), result['der']) for threshold, result in best['results'].items())
		axis.plot(*zip(*values), color=SERIES[index], marker=MARKERS[index], markersize=5, label=_name(run))

	axis.set_xlabel('AHC threshold')
	axis.set_ylabel('Validation DER (%)')
	if len(runs) > 1: axis.legend()
	else: axis.set_title(_name(runs[0]), color=TEXT)
	_save(figure, out, f"thresholds_{'_'.join(_name(run) for run in runs)}.pdf")


def _scalars(run, tag):
	accumulator = EventAccumulator(_folder(run), size_guidance={'scalars': 0})
	accumulator.Reload()

	return [event.value for event in accumulator.Scalars(tag)] if tag in accumulator.Tags()['scalars'] else []


def curves(runs, out):
	"""Per run: loss per epoch (train and validation) and validation DER per epoch, in separate panels."""
	for run in runs:
		figure, (loss_axis, der_axis) = plt.subplots(1, 2, figsize=(7, 2.6))

		for index, (tag, name) in enumerate([('loss/train', 'Train'), ('loss/val', 'Validation')]):
			values = _scalars(run, tag)
			loss_axis.plot(range(len(values)), values, color=SERIES[index], marker=MARKERS[index], markersize=5, label=name)

		loss_axis.set_title(f"Loss ({_config(run)['loss_fn']})", color=TEXT)
		loss_axis.set_xlabel('Epoch')
		loss_axis.legend()

		epochs = epoch_summaries(_folder(run))
		der_axis.plot([epoch['epoch'] for epoch in epochs], [epoch['der'] for epoch in epochs], color=SERIES[0], marker=MARKERS[0], markersize=5)
		der_axis.set_title('Validation DER (%)', color=TEXT)
		der_axis.set_xlabel('Epoch')

		figure.suptitle(_name(run), color=TEXT, fontsize=10, y=1.04)
		_save(figure, out, f'curves_{_name(run)}.pdf')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Figures for the results chapter')
	parser.add_argument('figure', choices=['histograms', 'thresholds', 'curves'])
	parser.add_argument('runs', nargs='+', help='Run IDs or run folders')
	parser.add_argument('--out', default=DEFAULT_OUT, help='Output folder for the PDFs')
	args = parser.parse_args()

	_style()
	{'histograms': histograms, 'thresholds': thresholds, 'curves': curves}[args.figure](args.runs, args.out)
