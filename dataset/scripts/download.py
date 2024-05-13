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

# If I ever need to manually download training data again.
# https://www.quora.com/How-can-you-download-files-exceeding-2-0-GB-on-Google-Drive

# curl -H "Authorization: Bearer ya29.a0AXooCgtSb6ZG-UNVZBxTmARduO47n1von-2RBy_C87GBoQWUsQJHMuHFZXTWZYOvIncXLQhwX8ARZwkRB8xZiKtjz6HEnRMF8Sj0ao4e4L7g6jHcLMMiB9hWVy_HWJsEENBlLBA6di9nCOaLM9Yzs1YXB_TzI8Onni81aCgYKAbUSARASFQHGX2MiJf6vujd0KERhvce_nJp7zg0171" https://www.googleapis.com/drive/v3/files/1j4aIdUvb8aQW0Jkzq7AcIBZhxs_nQT_w?alt=media -o train.zip