import time


class Timer:
	def __init__(self) -> None:
		self.start = time.time()
		self.TIME_FORMAT = '%02d:%02d:%06.3f'

	def get_time(self, format:str=None):
		current = time.time() - self.start

		return self._format_time(current, format)

	def reset(self):
		self.start = time.time()

	def _format_time(self, seconds:float, format:str=None):
		if format is None: format = self.TIME_FORMAT

		m, s = divmod(seconds, 60)
		h, m = divmod(m, 60)
		d, h = divmod(m, 24)

		days = f'{int(d)}d ' if d > 0 else ''

		return f'{days}{format % (h, m, s)}'