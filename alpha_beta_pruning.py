from time import time

class AlphaBeta:

	def __init__(self):
		self._terminal_state = []
		self._potential_state = []
		self._sub_potential_state = []
		self._full_state = (1 << 42) - 1
		self.cnt = 0

		for index in range(42):
			directions = ((0, 1), (1, 0), (1, -1), (1, 1))

			for direction in directions:
				row = index // 7
				col = index - row * 7
				state = 0
				indices = []

				for i in range(4):
					if row < 0 or row > 5 or col < 0 or col > 6:
						state = -1
						break
					state |= 1 << (row * 7 + col)
					indices.append(row * 7 + col)

					row += direction[0]
					col += direction[1]

				if state != -1:
					self._terminal_state.append(state)

					for i in indices:
						self._potential_state.append((state - (1 << i), 1 << i))

					for i in range(4):
						for j in range(i + 1, 4):
							self._sub_potential_state.append((state - (1 << indices[i]) - (1 << indices[j]), (1 << indices[i]) + (1 << indices[j])))

	def alpha_beta(self, state : (int, int), depth : int, alpha : int, beta : int, turn : bool, time_limit : float):
		if time() > time_limit:
			return 0
		if depth == 0 or self._is_terminal(state):
			return self._calculate_state(state)

		if turn:
			value = -10000
			for action in range(7):
				if (state[0] | state[1]) & (1 << action):
					continue
				next_state = self.get_next_state(state, action, turn)
				value = max(value, self.alpha_beta(next_state, depth - 1, alpha, beta, False, time_limit))
				if value >= beta:
					break
				alpha = max(alpha, value)
			return value
		else:
			value = 10000
			for action in range(7):
				if (state[0] | state[1]) & (1 << action):
					continue
				next_state = self.get_next_state(state, action, turn)
				value = min(value, self.alpha_beta(next_state, depth - 1, alpha, beta, True, time_limit))
				if value <= alpha:
					break
				beta = min(beta, value)
			return value

	def get_next_state(self, state : (int, int), action : int, turn : bool):
		for row in range(5, -1, -1):
			if (state[0] | state[1]) & (1 << (row * 7 + action)):
				continue
			if turn:
				state = (state[0] | (1 << (row * 7 + action)), state[1])
			else:
				state = (state[0], state[1] | (1 << (row * 7 + action)))
			return state


	def _calculate_state(self, state : (int, int)) -> int:
		for terminal in self._terminal_state:
			if (state[0] & terminal) == terminal:
				return 10000
			if (state[1] & terminal) == terminal:
				return -10000

		score = 0
		for potential in self._potential_state:
			if (state[0] & potential[0]) == potential[0] and not (state[1] & potential[1]):
				score += 5
			if (state[1] & potential[0]) == potential[0] and not (state[0] & potential[1]):
				score -= 5
		for potential in self._sub_potential_state:
			if (state[0] & potential[0]) == potential[0] and not (state[1] & potential[1]):
				score += 1
			if (state[1] & potential[0]) == potential[0] and not (state[0] & potential[1]):
				score -= 1
		return score

	def _is_terminal(self, state : (int, int)) -> bool:
		for terminal in self._terminal_state:
			if (state[0] & terminal) == terminal or (state[1] & terminal) == terminal:
				return True
		return False

	def print_state(self, state : (int, int)) -> None:

		print("+---+---+---+---+---+---+---+")
		for row in range(6):
			print('|', end = '')
			for col in range(7):
				shape = ' '
				index = row * 7 + col
				if state[0] & (1 << index):
					shape = 'O'
				if state[1] & (1 << index):
					shape = 'X'
				print(" {} |".format(shape), end = '')
			print("\n+---+---+---+---+---+---+---+")

def main():
	from time import sleep
	from random import choice
	alpha_beta = AlphaBeta()
	for state in alpha_beta._sub_potential_state:
		alpha_beta.print_state(state)
		sleep(0.4)

if __name__ == "__main__":
	main()