import numpy as np
from ReversiNode import *
from board import *
import random


def reverse_color(color):
	if color == 'X':
		return 'O'
	return 'X'


def match_color(color, winner):
	if color == 'X':
		return winner == 0
	else:
		return winner == 1


def sigmoid(x):
	return 1. / (1. + np.exp(-x))


def get_dis_score(board, color):
	cordinates_this = []
	cordinates_that = []
	for i in range(8):
		for j in range(8):
			if board[i][j] == color:
				cordinates_this.append([i, j])
			elif board[i][j] == reverse_color(color):
				cordinates_that.append([i, j])
	cordinates_this = np.array(cordinates_this, dtype=np.float)
	cordinates_that = np.array(cordinates_that, dtype=np.float)
	if len(cordinates_this) == 0:
		return 0
	if len(cordinates_that) == 0:
		return 1
	avg_this = np.average(cordinates_this, axis=0)
	avg_that = np.average(cordinates_that, axis=0)
	dis_this = np.sum(np.sqrt(np.sum((cordinates_this - avg_this) ** 2, axis=1))) / cordinates_this.shape[0]
	dis_that = np.sum(np.sqrt(np.sum((cordinates_that - avg_that) ** 2, axis=1))) / cordinates_that.shape[0]
	return dis_that - dis_this


def get_action_score(board, color):
	actions_this = list(board.get_legal_actions(color))
	actions_that = list(board.get_legal_actions(reverse_color(color)))
	return len(actions_this) - len(actions_that)


def get_stable_score(board, color):
	left = np.zeros((2, 8, 8))
	above = np.zeros((2, 8, 8))
	left_above = np.zeros((2, 8, 8))
	right_above = np.zeros((2, 8, 8))
	for i in range(8):
		for j in range(8):
			if board[i][j] == '.':
				continue
			k = 0 if board[i][j] == color else 1
			left[k][i][j] = 1 if (i == 0) or (i > 0 and left[k][i - 1][j]) else 0
			above[k][i][j] = 1 if (j == 0) or (j > 0 and above[k][i][j - 1]) else 0
			left_above[k][i][j] = 1 if (i == 0 or j == 0) or (i > 0 and j > 0 and left_above[k][i - 1][j - 1]) else 0
			right_above[k][i][j] = 1 if (i == 0 or j == 7) or (i > 0 and j < 7 and right_above[k][i - 1][j + 1]) else 0
	right = np.zeros((2, 8, 8))
	below = np.zeros((2, 8, 8))
	right_below = np.zeros((2, 8, 8))
	left_below = np.zeros((2, 8, 8))
	for i in reversed(range(8)):
		for j in reversed(range(8)):
			k = 0 if board[i][j] == color else 1
			right[k][i][j] = 1 if (i == 7) or (i < 7 and right[k][i + 1][j]) else 0
			below[k][i][j] = 1 if (j == 7) or (j < 7 and below[k][i][j + 1]) else 0
			right_below[k][i][j] = 1 if (i == 7 or j == 7) or (i < 7 and j < 7 and right_below[k][i + 1][j + 1]) else 0
			left_below[k][i][j] = 1 if (i == 7 or j == 0) or (i < 7 and j > 0 and left_below[k][i + 1][j - 1]) else 0
	left[left + right > 0] = 1
	above[above + below > 0] = 1
	left_above[left_above + right_below > 0] = 1
	right_above[right_above + left_below > 0] = 1
	stable = (left + above + left_above + right_above == 4)
	stable_this_cnt = np.sum(stable[0])
	stable_that_cnt = np.sum(stable[1])
	return stable_this_cnt - stable_that_cnt


def get_win_score(board, color):
	winner, score = board.get_winner()
	score = score if match_color(color, winner) else -score
	return score


def get_map_score(board, color):
	scores = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
						[-25, -45, 1, 1, 1, 1, -45, -25],
						[10, 1, 3, 2, 2, 3, 1, 10],
						[5, 1, 2, 1, 1, 2, 1, 5],
						[5, 1, 2, 1, 1, 2, 1, 5],
						[10, 1, 3, 2, 2, 3, 1, 10],
						[-25, -45, 1, 1, 1, 1, -45, -25],
						[500, -25, 10, 5, 5, 10, -25, 500]])
	score = 0
	for i in reversed(range(8)):
		for j in reversed(range(8)):
			if board[i][j] == color:
				score += scores[i][j]
	return score


class AIPlayer:
	"""
    AI 玩家
    """

	def __init__(self, color):
		"""
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

		self.color = color
		self.minimax_step = 3
		self.ab_step = 5
		self.mcts_n = 100
		self.search = 'alpha-beta'
		self.score = 0

	def get_move(self, board):
		"""
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
		if self.color == 'X':
			player_name = '黑棋'
		else:
			player_name = '白棋'
		print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

		# -----------------请实现你的算法代码--------------------------------------

		action = None
		if self.search == 'minimax':
			action, _ = self.minimax(board, self.minimax_step)
		elif self.search == 'mcts':
			root = ReversiNode(board=deepcopy(board), color=self.color)
			action = root.MCTS(self.mcts_n).from_action
		elif self.search == 'random':
			action = self.random_choice(board)
		else:
			action, _ = self.alpha_beta_prunig(board, self.ab_step)
		# ------------------------------------------------------------------------

		return action

	def random_choice(self, board):
		action_list = list(board.get_legal_actions(self.color))

		if len(action_list) == 0:
			return None
		else:
			return random.choice(action_list)

	def minimax(self, board, step):
		return self._minimax(board, self.color, step)

	def _minimax(self, board, color, step):
		action, val = None, None
		if step > 0:
			is_max_node = color == self.color
			legal_actions = list(board.get_legal_actions(color))
			if len(legal_actions):
				for action_t in legal_actions:
					flipped = board._move(action_t, color)
					_, val_t = self._minimax(board, reverse_color(color), step - 1)
					board.backpropagation(action_t, flipped, color)

					if val is None:
						val = val_t
						action = action_t
					else:
						if (is_max_node and (val_t > val)) or (not is_max_node and (val_t < val)):
							val = val_t
							action = action_t
			else:
				_, val = self._minimax(board, reverse_color(color), step - 1)
		else:
			val = self.get_score(board)
		return action, val

	def alpha_beta_prunig(self, board, step):
		import math
		return self._alpha_beta_prunig(board, self.color, step, -math.inf, math.inf)

	def _alpha_beta_prunig(self, board, color, step, alpha, beta):
		action, val = None, None
		if step > 0:
			is_max_node = color == self.color
			legal_actions = list(board.get_legal_actions(color))
			if len(legal_actions):
				for action_t in legal_actions:
					flipped = board._move(action_t, color)
					_, val_t = self._alpha_beta_prunig(board, reverse_color(color), step - 1, alpha, beta)
					board.backpropagation(action_t, flipped, color)
					if is_max_node and val_t > alpha:
						alpha = val_t
						action = action_t
					if not is_max_node and val_t < beta:
						beta = val_t
						action = action_t
					if alpha >= beta:
						break
				val = alpha if is_max_node else beta
			else:
				_, val = self._alpha_beta_prunig(board, reverse_color(color), step - 1, alpha, beta)
		else:
			val = self.get_score(board)
		return action, val

	def get_dis_score(self, board):
		return get_dis_score(board, self.color)

	def get_action_score(self, board):
		return get_action_score(board, self.color)

	def get_stable_score(self, board):
		return get_stable_score(board, self.color)

	def get_win_score(self, board):
		return get_win_score(board, self.color)

	def get_map_score(self, board):
		return get_map_score(board, self.color)

	def get_score(self, board):
		win_score = get_win_score(board, self.color)
		if self.score == 0:
			cnt = board.count(self.color) + board.count(reverse_color(self.color))
			scores = np.array([self.get_dis_score(board), self.get_action_score(board),
							   self.get_stable_score(board)])
			if cnt < 20:
				weights = np.array([5, 4, 1])
			elif cnt < 40:
				weights = np.array([1, 15, 3])
			else:
				weights = np.array([1, 1, 10])
			val = np.average(scores, weights=weights)
		else:
			val = win_score
		return val


if __name__ == '__main__':
	board = Board()
	a = [(0, i) for i in range(7)]
	a += [(1, i) for i in range(4)]
	a += [(2, i) for i in range(2)]
	a += [(3, i) for i in range(2)]
	a += [(4, 0), (4, 7)]
	a += [(5, 6 + i) for i in range(2)]
	a += [(6, 5 + i) for i in range(3)]
	a += [(7, 4 + i) for i in range(4)]
	for s in a:
		board[s[0]][s[1]] = 'X'
	board.display()
	print(get_stable_score(board, 'X'))
