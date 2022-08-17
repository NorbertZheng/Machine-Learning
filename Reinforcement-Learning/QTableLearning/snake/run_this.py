import time
from maze_env import Maze
from RL_brain import QLearningTable

def main():
	env = Maze()
	RL = QLearningTable(actions = list(range(env.n_actions)))

	for episode in range(100):
		if episode % 200 == 0:
			RL.save_q_table()
		# initial observation
		observation = env.reset()
		counter = 0

		while True:
			# fresh env
			env.render()
			print("Round: " + str(counter))

			# RL choose action based on observation
			action = RL.choose_action(observation)

			# RL take action and get next observation & reward
			observation_, reward, done = env.step(action)

			# RL learn from this transition
			RL.learn(observation, action, reward, observation_, done)

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				# RL.save_q_table()
				break
			else:
				time.sleep(1)
				counter += 1

		# end game
		print("end game")

	# save q_table
	RL.save_q_table()

if __name__ == '__main__':
	main()

