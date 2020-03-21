from board import *


def reverse_color(color):
	if color == 'X':
		return 'O'
	return 'X'

def match_color(color, win):
	if color == 'X':
		return win == 0
	else:
		return win == 1


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
		self.ab_step = 4
		self.search = 'alpha-beta'

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
			action, val = self.minimax(board, self.color, self.minimax_step)
		else:
			import math
			action, val = self.alpha_beta_prunig(board, self.color, self.ab_step, -math.inf, math.inf)
		# ------------------------------------------------------------------------

		return action
	
	def get_score(self, board):
		win, val = board.get_winner()
		if not match_color(self.color, win):
			val = -val
		return val

	def minimax(self, board, color, step):
		action, val = None, None
		if step > 0:
			is_max_node = color == self.color
			legal_actions = list(board.get_legal_actions(color))
			if len(legal_actions):
				for action_t in legal_actions:
					flipped = board._move(action_t, color)
					_, val_t = self.minimax(board, reverse_color(color), step-1)
					board.backpropagation(action_t, flipped, color)

					if val is None:
						val = val_t
						action = action_t
					else:
						if (is_max_node and (val_t > val)) or (not is_max_node and (val_t < val)):
							val = val_t
							action = action_t
			else:
				_, val = self.minimax(board, reverse_color(color), step-1)
		else:
			val = self.get_score(board)
		return action, val

	def alpha_beta_prunig(self, board, color, step, alpha, beta):
		action, val = None, None
		if step > 0:
			is_max_node = color == self.color
			legal_actions = list(board.get_legal_actions(color))
			if len(legal_actions):
				for action_t in legal_actions:
					flipped = board._move(action_t, color)
					_, val_t = self.alpha_beta_prunig(board, reverse_color(color), step-1, alpha, beta)
					board.backpropagation(action_t, flipped, color)

					if is_max_node and val_t > alpha:
						alpha = val_t
						action = action_t
					if not is_max_node and val_t < beta:
						beta = val_t
						action = action_t
					if alpha >= beta:
						break
				if is_max_node:
					val = alpha
				else:
					val = beta
			else:
				_, val = self.alpha_beta_prunig(board, reverse_color(color), step-1, alpha, beta)
		else:
			val = self.get_score(board)
		return action, val