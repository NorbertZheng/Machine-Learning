import os
import time
import random
import numpy as np

class Maze:
	n_actions = 4

	def __init__(self, _width = 5, _length = 3):
		# flags
		self.FREE_FLAG = 0
		self.HEAD_FLAG = 1
		self.BODY_FLAG = 2
		self.FOOD_FLAG = 3
		# reward
		self.GET_NONE = 0
		self.GET_FOOD = 5
		self.GET_BODY = -30
		# meta attr
		self.width = _width
		self.field = np.zeros((self.width, self.width), dtype = np.int32)
		self._length = _length
		self.length = _length
		self.direction = []
		# init field
		self._init_field()

	def reset(self):
		self.length = self._length
		self.direction = []
		# init field
		self._init_field()
		return (self.field.copy(), self.direction[0])

	def _reset_field(self):
		self.field = np.zeros((self.width, self.width), dtype = np.int32)

	def _init_field(self):
		self._reset_field()
		if self.length < 1:
			self.length = 1
		# random snake head
		row = random.randint(0, np.shape(self.field)[0] - 1)
		col = random.randint(0, np.shape(self.field)[1] - 1)
		self.field[row, col] = self.HEAD_FLAG
		# generate body
		body_length = self.length - 1
		while body_length > 0:
			# get free loc
			n_free_loc = [0, 0, 0, 0]
			if (row > 0) and (self.field[row - 1, col] == self.FREE_FLAG):
				n_free_loc[0] = 1
			if (col < np.shape(self.field)[1] - 1) and (self.field[row, col + 1] == self.FREE_FLAG):
				n_free_loc[1] = 1
			if (row < np.shape(self.field)[0] - 1) and (self.field[row + 1, col] == self.FREE_FLAG):
				n_free_loc[2] = 1
			if (col > 0) and (self.field[row, col - 1] == self.FREE_FLAG):
				n_free_loc[3] = 1
			# none is available
			if sum(n_free_loc) == 0:
				self.length -= body_length
				break
			else:
				body_loc = random.randint(0, sum(n_free_loc) - 1)
				for i in range(len(n_free_loc)):
					if (n_free_loc[i] == 0):
						body_loc += 1
					elif (i == body_loc):
						# find body_loc
						if body_loc == 0:
							self.field[row - 1, col] = self.BODY_FLAG
							row = row - 1
							self.direction.append(2)
							break
						elif body_loc == 1:
							self.field[row, col + 1] = self.BODY_FLAG
							col = col + 1
							self.direction.append(3)
							break
						elif body_loc == 2:
							self.field[row + 1, col] = self.BODY_FLAG
							row = row + 1
							self.direction.append(0)
							break
						elif body_loc == 3:
							self.field[row, col - 1] = self.BODY_FLAG
							col = col - 1
							self.direction.append(1)
							break
						else:
							print("ERROR: (Maze._init_firld) unknown body_loc!")
							break
			# decrease length
			body_length -= 1
		# generate food
		self._generate_food()

	def _generate_food(self):
		row = random.randint(0, np.shape(self.field)[0] - 1)
		col = random.randint(0, np.shape(self.field)[1] - 1)
		# generate food
		while self.field[row, col] != self.FREE_FLAG:
			row = random.randint(0, np.shape(self.field)[0] - 1)
			col = random.randint(0, np.shape(self.field)[1] - 1)
		self.field[row, col] = self.FOOD_FLAG

	def render(self):
		# clear
		os.system("clear")
		# title
		print("\t\t\tSnake with RL")
		# up
		line = []
		for i in range(np.shape(self.field)[1] + 2):
			line.append("-")
		print("\t\t\t" + "".join(line))
		# field
		for i in range(np.shape(self.field)[0]):
			line = []
			for j in range(np.shape(self.field)[1]):
				if self.field[i, j] == self.FREE_FLAG:
					line.append(" ")
				elif self.field[i, j] == self.HEAD_FLAG:
					line.append("@")
				elif self.field[i, j] == self.BODY_FLAG:
					line.append("*")
				elif self.field[i, j] == self.FOOD_FLAG:
					line.append("$")
				else:
					print("ERROR: (Maze.show_field) unknown flag!")
					break
			print("\t\t\t|" + "".join(line) + "|")
			# bottom
		line = []
		for i in range(np.shape(self.field)[1] + 2):
			line.append("-")
		print("\t\t\t" + "".join(line))
		print("\tdirection: " + str(self.direction) + "\thead_loc: " + str(self._get_head()) + "\ttail_loc: " + str(self._get_tail()))

	def _get_head(self):
		for i in range(np.shape(self.field)[0]):
			for j in range(np.shape(self.field)[1]):
				if self.field[i, j] == self.HEAD_FLAG:
					return (i, j)
		return (np.inf, np.inf)

	def _get_tail(self):
		row, col = self._get_head()
		if row == np.inf:
			return (row, col)
		for i in range(len(self.direction)):
			if self.direction[i] == 0:
				row = (row + 1) % np.shape(self.field)[0]
			elif self.direction[i] == 1:
				col = (col - 1) % np.shape(self.field)[1]
			elif self.direction[i] == 2:
				row = (row - 1) % np.shape(self.field)[0]
			elif self.direction[i] == 3:
				col = (col + 1) % np.shape(self.field)[1]
			else:
				print("ERROR: (Maze._get_tail) unknown direction!")
		return (row, col)

	def _all_filled(self):
		for i in range(np.shape(self.field)[0]):
			for j in range(np.shape(self.field)[1]):
				if self.field[i, j] == self.FREE_FLAG:
					return False
		return True

	def step(self, action):
		# print(type(action))
		if (action + 2) % 4 == self.direction[0]:
			print("ERROR: (Maze.step) forbidden action!")
			return ((self.field.copy(), self.direction[0]), self.GET_NONE, True)
		else:
			head_row, head_col = self._get_head()
			tail_row, tail_col = self._get_tail()
			if action == 0:
				if (self.field[(head_row - 1) % np.shape(self.field)[0], head_col] == self.FOOD_FLAG):
					self.field[(head_row - 1) % np.shape(self.field)[0], head_col] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction.insert(0, action)
					if self._all_filled():
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, True)
					else:
						self._generate_food()
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, False)
				elif ((self.field[(head_row - 1) % np.shape(self.field)[0], head_col] == self.BODY_FLAG) and ((((head_row - 1) % np.shape(self.field)[0]) != tail_row) or (head_col != tail_col))):
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[(head_row - 1) % np.shape(self.field)[0], head_col] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_BODY, True)
				else:
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[(head_row - 1) % np.shape(self.field)[0], head_col] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_NONE, self._all_filled())
			elif action == 1:
				if (self.field[head_row, (head_col + 1) % np.shape(self.field)[1]] == self.FOOD_FLAG):
					self.field[head_row, (head_col + 1) % np.shape(self.field)[1]] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction.insert(0, action)
					if self._all_filled():
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, True)
					else:
						self._generate_food()
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, False)
				elif ((self.field[head_row, (head_col + 1) % np.shape(self.field)[1]] == self.BODY_FLAG) and ((head_row != tail_row) or (((head_col + 1) % np.shape(self.field)[1]) != tail_col))):
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[head_row, (head_col + 1) % np.shape(self.field)[1]] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_BODY, True)
				else:
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[head_row, (head_col + 1) % np.shape(self.field)[1]] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_NONE, self._all_filled())
			elif action == 2:
				if (self.field[(head_row + 1) % np.shape(self.field)[0], head_col] == self.FOOD_FLAG):
					self.field[(head_row + 1) % np.shape(self.field)[0], head_col] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction.insert(0, action)
					if self._all_filled():
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, True)
					else:
						self._generate_food()
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, False)
				elif ((self.field[(head_row + 1) % np.shape(self.field)[0], head_col] == self.BODY_FLAG) and ((((head_row + 1) % np.shape(self.field)[0]) != tail_row) or (head_col != tail_col))):
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[(head_row + 1) % np.shape(self.field)[0], head_col] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_BODY, True)
				else:
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[(head_row + 1) % np.shape(self.field)[0], head_col] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_NONE, self._all_filled())
			elif action == 3:
				if (self.field[head_row, (head_col - 1) % np.shape(self.field)[1]] == self.FOOD_FLAG):
					self.field[head_row, (head_col - 1) % np.shape(self.field)[1]] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction.insert(0, action)
					if self._all_filled():
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, True)
					else:
						self._generate_food()
						return ((self.field.copy(), self.direction[0]), self.GET_FOOD, False)
				elif ((self.field[head_row, (head_col - 1) % np.shape(self.field)[1]] == self.BODY_FLAG) and ((head_row != tail_row) or (((head_col - 1) % np.shape(self.field)[1]) != tail_col))):
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[head_row, (head_col - 1) % np.shape(self.field)[1]] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_BODY, True)
				else:
					self.field[tail_row, tail_col] = self.FREE_FLAG
					self.field[head_row, (head_col - 1) % np.shape(self.field)[1]] = self.HEAD_FLAG
					self.field[head_row, head_col] = self.BODY_FLAG
					# update direction
					self.direction[1:] = self.direction[:-1]
					self.direction[0] = action
					return ((self.field.copy(), self.direction[0]), self.GET_NONE, self._all_filled())
			else:
				print("ERROR: (Maze.step) unknown action!")
				return ((self.field.copy(), self.direction[0]), self.GET_NONE, True)


if __name__ == '__main__':
	env = Maze()
	env.render()
	env.step(0)
	time.sleep(2)
	env.render()
	env.step(1)
	time.sleep(2)
	env.render()
	env.step(2)
	time.sleep(2)
	env.render()
	env.step(3)
	time.sleep(2)
	env.render()

					
