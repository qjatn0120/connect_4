import numpy as np

def next_state(state : np.ndarray[6, 7], action : int, player : int):
	state = np.copy(state)
	row = np.max(np.where(state[:, action] == 0)[0])
	state[row][action] = player
	return state

def check_equal(state : np.ndarray[6, 7], indices: tuple):
	res = 0
	for index in indices:
		if index[0] < 0 or index[0] > 5 or index[1] < 0 or index[1] > 6:
			return 0

		val = state[index[0]][index[1]]
		if val == 0:
			return 0
		
		if res == 0:
			res = val
		elif res != val:
			return 0
		res = val
	return res