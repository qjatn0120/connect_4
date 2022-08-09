from abc import *
import numpy as np
from random import Random
from config import *
from utils import next_state
from alpha_beta_pruning import AlphaBeta

class Agent(metaclass = ABCMeta):
	@abstractmethod
	def action(self, state : np.ndarray[6, 7]) -> int:
		pass

# LEVEL 0
# win / lose / draw (100 games)
# vs LEVEL 1 0 / 100 / 0
# vs LEVEL 2 0 / 100 / 0
class RandomAgent(Agent):
	def __init__(self, seed : int = AGENT_SEED):
		self.random = Random(seed)

	def action(self, state : np.ndarray[6, 7]) -> int:
		possible_action = np.where(state[0] == 0)
		return self.random.choice(possible_action[0]).item()

# LEVEL 1
# win / lose / draw (100 games)
# vs LEVEL 0 100 / 0 / 0
# vs LEVEL 2 19 / 79 / 2
class GreedyAgent(Agent):
	def __init__(self, seed : int = AGENT_SEED):
		self._random = Random(seed)
		self._score_table = GREEDY_SCORE_TABLE

	def action(self, state : np.ndarray[6, 7]) -> int:
		score = []
		for action in range(7):
			if state[0][action]:
				score.append(-100000)
				continue
			temp_list = []
			temp_state = next_state(state, action, 1)
			if self._calculate_state(temp_state) > 9000:
				return action
			for op_action in range(7):
				if temp_state[0][op_action]:
					temp_list.append(100000)
					continue
				temp_list.append(self._calculate_state(next_state(temp_state, op_action, -1)))
			score.append(min(temp_list))
		score = np.array(score)
		return self._random.choice(np.where(score == np.max(score))[0]).item()

	def _calculate_state(self, state : np.ndarray[6, 7]) -> int:
		score = 0
		for index in range(42):
			row = index // 7
			col = index - row * 7
			score += self._check_score(state, row, col, 1, 0)
			score += self._check_score(state, row, col, 0, 1)
			score += self._check_score(state, row, col, 1, -1)
			score += self._check_score(state, row, col, 1, 1)
		return score

	def _check_score(self, state : np.ndarray[6, 7], row : int, col : int, rowdir : int, coldir : int) -> int:
		playerCount, oppnentCount = 0, 0
		for i in range(4):
			if row < 0 or row > 5 or col < 0 or col > 6:
				return 0
			val = state[row][col]
			if val == 1:
				oppnentCount = -100
				playerCount += 1
			if val == -1:
				playerCount = -100
				oppnentCount += 1
			row += rowdir
			col += coldir

		if playerCount < 0 and oppnentCount < 0:
			return 0
		if oppnentCount < 0:
			return self._score_table[playerCount]
		return -self._score_table[oppnentCount]

# LEVEL 2
# win / lose / draw (100 games)
# vs LEVEL 0 100 / 0 / 0
# vs LEVEL 1 79 / 19 / 2
class AlphaBetaAgent(Agent):
	def __init__(self, seed : int = AGENT_SEED):
		self._random = Random(seed)
		self._score_table = GREEDY_SCORE_TABLE
		self.alpha_beta = AlphaBeta()

	def action(self, state : np.ndarray[6, 7]) -> int:
		state0, state1 = 0, 0
		for index in range(42):
			row = index // 7
			col = index - row * 7
			if state[row][col] == 1:
				state0 |= (1 << index)
			if state[row][col] == -1:
				state1 |= (1 << index)
		state = (state0, state1)

		score = []
		for action in range(7):
			if (state[0] | state[1]) & (1 << action):
				score.append(-100000)
				continue
			target_state = self.alpha_beta.get_next_state(state, action, True)
			score.append(self.alpha_beta.alpha_beta(target_state, 5, -10000, 10000, False))

		score = np.array(score)
		return self._random.choice(np.where(score == np.max(score))[0]).item()


def main():
	from game import Connect4
	from random import choice
	from time import sleep

	player = AlphaBetaAgent()
	opponent = GreedyAgent()
	env = Connect4(opponent)
	state = env.reset(True)
	done = False

	while not done:
		action = player.action(state)
		try:
			state, reward, done, _ = env.step(action)
		except Exception as e:
			break
		env.render()

	if reward == 1:
		print("PLAYER WIN")
	elif reward == -1:
		print("OPPONENT WIN")
	else:
		print("DRAW")

if __name__ == "__main__":
	main()