class InvalidType(Exception):
	def __init__(self, given, expected):
		super().__init__('Given type is {}. Expected type is {}'.format(given, expected))

class RowOutofRangeException(Exception):
	def __init__(self):
		super().__init__('Given row is out of range. It should be between 0 and 6.')

class FullRowException(Exception):
	def __init__(self):
		super().__init__('Given row is full. Try another row.')

class ActionTimeOutException(Exception):
	def __init__(self, used_time, expected_time):
		super().__init__('Timeout. Given time is {} second. You used {} second'.format(expected_time, used_time))