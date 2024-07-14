import torch


class TrainCollator:
	def __call__(self, batch):
		result = []
		for group in batch:
			flat_batch = flatten_list(group)

			result.append(add_fields(flat_batch))

		result = add_fields(result)

		return result


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

	for field in fields:
		if isinstance(data_list[0][field], torch.Tensor):
			size = (len(data_list), *data_list[0][field].size())
			result[field] = data_list[0][field].new_empty(size)
		else:
			result[field] = [None for _ in range(len(data_list))]

		for index, sample in enumerate(data_list):
			result[field][index] = sample[field]

		if isinstance(data_list[0][field], dict):
			result[field] = load_dict(result[field])

	return result


def load_dict(data):
	result = {}

	for key in data[0].keys():
		result[key] = []

	for element in data:
		for key, value in element.items():
			result[key].append(value)

	return result