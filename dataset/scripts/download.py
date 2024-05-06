import gdown, zipfile, os


def main():
	ids = {
		'train':	'1j4aIdUvb8aQW0Jkzq7AcIBZhxs_nQT_w&confirm=t',
		'val': 		'1acHflaGbeEscqDF8gSpnxWHn3-q4Mye-&confirm=t',
		'test': 	'11YWpZGM7XSwYtqWeg-DzhxFANYwsY3qo&confirm=t',
	}

	for dataset in ['train', 'val', 'test']:
		zip_path = f'dataset/{dataset}.zip'

		gdown.download(id=ids[dataset], output=zip_path)

		with zipfile.ZipFile(zip_path, 'r') as file:
			file.extractall('dataset')

		os.remove(zip_path)


if __name__ == '__main__':
	main()