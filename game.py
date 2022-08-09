import numpy as np
from exception import *
from agent import Agent, RandomAgent
from utils import check_equal

class Connect4:

	def __init__(self, opponent_agent : Agent):
		self._agent = opponent_agent

	def reset(self, first_turn = True) -> np.ndarray[6, 7]:
		self._board = np.zeros((6, 7), dtype = int)
		self._cell_left = 42
		if not first_turn:
			opponent_action = self._agent.action(np.copy(self._board) * -1)
			row_index = np.max(np.where(self._board[:, opponent_action] == 0)[0])
			self._board[row_index][opponent_action] = -1
			self._cell_left -= 1
		return self._board

	def step(self, action : int) -> [np.ndarray[6, 7], int, bool, dict]:
		try:
			if type(action) != int:
				raise InvalidType(type(action).__name__, "int")
		except Exception as e:
			print(e)
			return state, -1, True, {}

		try:
			if action < 0 or action > 6:
				raise RowOutofRangeException()
		except Exception as e:
			print('Given Row is', action)
			print(e)
			return state, -1, True, {}
		
		column = self._board[:, action]
		column = np.where(column == 0)[0]
		
		try:
			if column.size == 0:
				raise FullRowException()
		except Exception as e:
			print(e)
			return state, -1, True, {}

		row_index = np.max(column)
		self._board[row_index][action] = 1
		self._cell_left -= 1
		reward = self._check_reward()
		done = reward != 0 or self._cell_left == 0

		if not done:
			opponent_action = self._agent.action(np.copy(self._board) * -1)
			row_index = np.max(np.where(self._board[:, opponent_action] == 0)[0])
			self._board[row_index][opponent_action] = -1
			self._cell_left -= 1
			reward = self._check_reward()
			done = reward != 0 or self._cell_left == 0
		info = {}

		return np.copy(self._board), reward, done, info

	# return whether game is finished or not
	# return 1 if player win, -1 if opponent win and 0 if game is not finished
	def _check_reward(self) -> int:
		for index in range(42):
			row = index // 7
			col = index - row * 7
			res = check_equal(self._board, ((row, col), (row, col + 1), (row, col + 2), (row, col + 3)))
			if res:
				return res
			res = check_equal(self._board, ((row, col), (row + 1, col), (row + 2, col), (row + 3, col)))
			if res:
				return res
			res = check_equal(self._board, ((row, col), (row + 1, col - 1), (row + 2, col - 2), (row + 3, col - 3)))
			if res:
				return res
			res = check_equal(self._board, ((row, col), (row + 1, col + 1), (row + 2, col + 2), (row + 3, col + 3)))
			if res:
				return res
		return 0

	def render(self) -> None:
		print("+---+---+---+---+---+---+---+")
		print("| 0 | 1 | 2 | 3 | 4 | 5 | 6 |")
		print("+---+---+---+---+---+---+---+")
		for row in range(6):
			print('|', end = '')
			for col in range(7):
				shape = ' '
				if self._board[row][col] == 1:
					shape = 'O'
				if self._board[row][col] == -1:
					shape = 'X'
				print(" {} |".format(shape), end = '')
			print("\n+---+---+---+---+---+---+---+")

def main():
	opponent_agent = RandomAgent()
	env = Connect4(opponent_agent)
	env.reset(False)
	env.step(0)
	env.render()

if __name__ == "__main__":
	main()