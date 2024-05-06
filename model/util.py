from multiprocessing import Pool

def cpu_parallel_process(function, collection, threads, args):
	collection_split = []
	collection_size = len(collection)

	for i in range(threads):
		collection_split.append(collection[i:collection_size:threads])

	pool = Pool(threads)
	tasks = [(sublist, args) for sublist in collection_split]

	pool.map(function, tasks)
	pool.close()
	pool.join()


def intersection_over_union(boxA, boxB, relativeToBoxA=False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	if relativeToBoxA == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou


def argparse_helper(parser, **kwargs):
	if kwargs:
		args_list = []

		for key, value in kwargs.items():
			if key == 'not_empty': continue

			args_list.append(f'--{key}')
			if isinstance(value, str): args_list.append(value)
	else:
		args_list = None

	args = parser.parse_args(args=args_list)

	return args