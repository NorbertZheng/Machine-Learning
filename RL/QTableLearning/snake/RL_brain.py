import os
import time
import pickle
import numpy as np
import pandas as pd

class QLearningTable:
	Q_TABLE_CSV = "./q_table.csv"

	def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
		self.actions = actions		# a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		if os.path.exists(QLearningTable.Q_TABLE_CSV):
			# self.q_table = pd.read_csv(QLearningTable.Q_TABLE_CSV, index_col = 0, header = 0)
			with open(QLearningTable.Q_TABLE_CSV, "rb") as f:
				self.q_table = pickle.load(f)
		else:
			self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

	def choose_action(self, observation):
		state = self._nparray_to_str(observation)
		forbidden_direction = (observation[1] + 2) % 4
		self.check_state_exist(state)
		# print(self.q_table)
		# action selection
		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.loc[state, :]
			# some actions may have the same value, randomly choose on in these actions
			action = np.random.choice(state_action[state_action == np.max(state_action)].index)
			# check valid, random
			if action == forbidden_direction:
				print("WARNING: (RL_brain.choose_action) action == forbidden_direction!")
			while action == forbidden_direction:
				action = np.random.choice(self.actions)	
		else:
 			# choose random action
			action = np.random.choice(self.actions)
			while action == forbidden_direction:
				action = np.random.choice(self.actions)
		return int(action)

	def learn(self, s, a, r, s_, done):
		s = self._nparray_to_str(s)
		s_ = self._nparray_to_str(s_)
		self.check_state_exist(s_)
		# print(self.q_table)
		q_predict = self.q_table.loc[s, a]
		if not done:
			q_target = r + self.gamma * self.q_table.loc[s_, :].max()	# next state is not terminal
		else:
			q_target = r		# next state is terminal
		self.q_table.loc[s, a] += self.lr * (q_target - q_predict)		# update

	def save_q_table(self):
		# self.q_table.to_csv(QLearningTable.Q_TABLE_CSV, encoding = "utf-8-sig", header = True, index = True)
		with open(QLearningTable.Q_TABLE_CSV, "wb") as f:
			pickle.dump(self.q_table, f)
		print(self.q_table)
		time.sleep(1)

	def _nparray_to_str(self, observation):
		state = ""
		field, direction = observation[0], observation[1]
		state += str(direction)
		for i in range(np.shape(field)[0]):
			for j in range(np.shape(field)[1]):
				state += str(field[i, j])
		return state

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series(
					[0] * len(self.actions),
					index = self.q_table.columns,
					name = state,
				)
			)
