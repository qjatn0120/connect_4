from game import Connect4
from agent import RandomAgent, GreedyAgent

def main():
	env = Connect4(GreedyAgent())
	state = env.reset()
	done = False
	while not done:
		env.render()
		print("Enter column to place : ", end = '')
		action = int(input())
		state, reward, done, _ = env.step(action)

	env.render()
	if reward == -1:
		print("YOU LOSE")
	elif reward == 1:
		print("YOU WIN")
	else:
		print("DRAW")
if __name__ == "__main__":
	main()