from game import Connect4
from agent import *

def test_agent(test_count : int, target_agent : Agent, opponent_agent : Agent):

	env = Connect4(opponent_agent)
	win_count = 0
	lose_count = 0

	for count in range(test_count):
		state = env.reset(test_count % 2 == 0)
		done = False

		while not done:
			action = target_agent.action(state)
			try:
				state, reward, done, _ = env.step(action)
			except Exception as e:
				break

		print("Game [{} / {}]".format(count + 1, test_count), end = ' ')
		if reward == 1:
			print("PLAYER WIN")
			win_count += 1
		elif reward == -1:
			print("OPPONENT WIN")
			lose_count += 1
		else:
			print("DRAW")

	print("win / lose : {} / {}".format(win_count, lose_count))

alpha_beta = AlphaBetaAgent()
greedy = GreedyAgent()
test_agent(100, alpha_beta, greedy)