# Experiment registry (see CHECKLIST.md, section 7a, in the thesis document repository).
# Every run starts from the same point: relation layer from scratch, no checkpoint, same seed and epochs.
# Logs and checkpoints of each run go to model/third_party/avr_net/checkpoints/lightning_logs/<RUN_ID>.

BASE = {
	# Reproducibility
	'seed':						42,
	'epochs':					10,
	'relation_init':			None,		# None: relation layer from scratch
	'checkpoint':				None,

	# Data config
	'video_proportion': 		1.0,
	'val_video_proportion':		1.0,
	'aligned': 					False,
	'balanced':					True,
	'training':					'pairs',	# 'pairs': balanced pairs of each video. 'episodic': episodes of the original AVR-Net (E0)
	'shuffle_block_videos':		0,			# 0: preload train and shuffle every pair. N: shuffle within blocks of N videos
	'max_frames': 				60,
	'frame_selection':			'first',	# 'first' or 'center' max_frames of each utterance
	'db_video_mode': 			'average',	# 'pick_first' 'pick_random' 'keep_all' 'average'
	'task':						'train',	# 'train' 'val' 'test'

	# Architecture
	'self_attention': 			'pick_first', 	# 'class_token', 'temporal', 'pick_first'
	'self_attention_dropout':	0.2,
	'cross_attention':	 		'concat', 		# 'fusion' 'concat'
	'fine_tunning':				False,

	# Hyperparams
	'loss_fn':					'contrastive',	# 'bce' 'mse' 'contrastive'
	'pos_margin':				0.40,			# Only used by contrastive loss
	'neg_margin':				0.80, 			# Only used by contrastive loss
	'ahc_threshold':			0.40,
	'optimizer':				'sgd', 			# 'sgd' 'adam'
	'learning_rate': 			5e-4,
	'momentum': 				0.0,
	'weight_decay': 			1e-4,
	'step_size': 				1,
	'gamma': 					0.99,
	'frozen_epochs': 			0,
	'disable_pb': 				False,
}

# Pilots: short runs on a fraction of the data to check that a configuration learns from scratch before the full runs.
# The first full E4 (SGD, lr 5e-4, no momentum, unshuffled) stayed at chance level for 8 epochs (CHECKLIST.md, section 7a).
PILOT = {'video_proportion': 0.1, 'val_video_proportion': 0.25, 'epochs': 3}

# Only the differences with BASE
RUNS = {
	'P1':		{**PILOT},																	# E4 as it was, now shuffled
	'P2':		{**PILOT, 'learning_rate': 1e-3,	'momentum': 0.9},
	'P3':		{**PILOT, 'learning_rate': 1e-2,	'momentum': 0.9},
	'P4':		{**PILOT, 'optimizer': 'adam'},												# E3: contrastive + Adam
	'P5':		{**PILOT, 'optimizer': 'adam',		'loss_fn': 'mse'},						# E2: original AVR-Net loss and optimizer

	# Proposed model. Run it first: it is also the pilot (E-PILOTO), it must reach a DER close to the previous results
	'E4':		{},

	# Incremental chain: baseline -> + full dataset -> + average -> + contrastive -> + SGD (E4)
	# E0: same as E1 except for which pairs it sees: pairs of new episodes every epoch, as many as E1, in batches of 64 (DECISIONES.md R3.2)
	'E0':		{'loss_fn': 'mse',			'optimizer': 'adam', 'db_video_mode': 'pick_random', 'training': 'episodic', 'balanced': False},
	'E1':		{'loss_fn': 'mse',			'optimizer': 'adam', 'db_video_mode': 'pick_random'},
	'E2':		{'loss_fn': 'mse',			'optimizer': 'adam'},
	'E3':		{'loss_fn': 'contrastive',	'optimizer': 'adam'},

	# Loss x optimizer table
	'E5':		{'loss_fn': 'bce',			'optimizer': 'adam'},
	'E6':		{'loss_fn': 'bce',			'optimizer': 'sgd'},
	'E7':		{'loss_fn': 'mse',			'optimizer': 'sgd'},

	# Attention (negative results)
	# keep_all with 60 frames does not fit on disk: the 10 central frames of each utterance, shuffled in blocks of 4 videos
	# because they do not fit in memory either (DECISIONES.md R2.2)
	'E8':		{'db_video_mode': 'keep_all', 'self_attention': 'class_token', 'max_frames': 10, 'frame_selection': 'center', 'shuffle_block_videos': 4},
	'E9':		{'db_video_mode': 'keep_all', 'self_attention': 'temporal', 'max_frames': 10, 'frame_selection': 'center', 'shuffle_block_videos': 4},
	'E10':		{'cross_attention': 'fusion'},

	# E13: best configuration with faces replaced at random, p = 0.5 as in the AVR-Net paper. Pending: missing_rate is not
	# implemented (tools/dataset.py) and the best configuration is not chosen yet
}


def get_run(run_id):
	assert run_id in RUNS, f"Unknown run '{run_id}'. Available runs: {', '.join(RUNS)}"

	return {**BASE, **RUNS[run_id], 'run_name': run_id}
