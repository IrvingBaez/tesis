import torch
import torch.nn.functional as F


# TODO: Split into feature collator and clustering collator
class CustomCollator:
	def __call__(self, batch):
		flat_batch = flatten_list(batch)
		new_batch = add_fields(flat_batch)

		return new_batch


def flatten_list(nested_list):
	flat_list = []

	for element in nested_list:
		if isinstance(element, list):
			flat_list.extend(flatten_list(element))
		else:
			flat_list.append(element)

	return flat_list


def add_fields(data_list):
	result = {}
	fields = data_list[0].keys()
	chunk_size = len(data_list)

	# TODO: Cumulate tensors in CPU, then delete og tensors and move cummulative to GPU (in corresponding device).
	for field in fields:
		if isinstance(data_list[0][field], torch.Tensor):
			size = (chunk_size, *data_list[0][field].size())
			result[field] = data_list[0][field].new_empty(size)
		else:
			result[field] = [None for _ in range(chunk_size)]

		for index, sample in enumerate(data_list):
			result[field][index] = sample[field]

		if isinstance(data_list[0][field], dict):
			result[field] = load_dict(result[field])

	# For feature extraction:
	if isinstance(result['audio'], torch.Tensor) and result['audio'].numel() == chunk_size * 1 * 32000:
		result['audio'] = result['audio'].reshape((chunk_size, 1, 32000))

	if 'frames' in result.keys():
		batch_size, channels, frames, height, width = result['frames'].shape
		result['frames'] = result['frames'].reshape((batch_size, frames, channels, 1, height, width))

	# For clustering
	if 'task_full' in result.keys():
		result['task_full'] = torch.tensor(result['task_full'])

	return result


def load_dict(data):
	result = {}

	for key in data[0].keys():
		result[key] = []

	for element in data:
		for key, value in element.items():
			result[key].append(value)

	result['start'] = torch.tensor(result['start'])
	result['end'] = torch.tensor(result['end'])

	return result