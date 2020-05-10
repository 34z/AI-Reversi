import math
import abc


class MCTreeNode:
	def __init__(self, parent=None):
		self.parent = parent
		self.children = []
		self.reward = 0
		self.visit = 0
		self.ucb_c = math.sqrt(1 / 2)
		self.select_func = None
		self.backup_func = None
	
	@property
	def n_ch(self):
		return len(self.children)

	@property
	def is_root(self):
		return self.parent is None

	@abc.abstractproperty
	def is_leaf(self):
		pass

	@abc.abstractmethod
	def is_fully_expanded(self):
		pass
	
	def tree_policy(self):
		if self.is_leaf:
			return self
		if self.is_fully_expanded():
			best, _ = self.best_child('ucb')
			if self.select_func:
				self.select_func(best)
			return best.tree_policy()
		else:
			return self.expand()

	@abc.abstractmethod
	def expand(self):
		pass

	def cal_prob(self):
		return self.reward / self.visit

	def cal_ucb(self):
		return self.cal_prob() + self.ucb_c * math.sqrt(2 * math.log1p(self.parent.visit) / self.visit)

	def best_child(self, mode=None):
		best = None
		val = None
		for child in self.children:
			if mode == 'ucb':
				val_t = child.cal_ucb()
			else:
				val_t = self.cal_prob()
			if best is None or val_t > val:
				best = child
				val = val_t
		return best, val

	@abc.abstractmethod
	def default_policy(self):
		pass

	def backup(self, reward):
		self.reward += reward
		self.visit += 1
		if self.backup_func:
			self.backup_func()
		if not self.is_root:
			self.parent.backup(reward)
	
	def MCTS(self, n):
		assert self.is_root
		while n:
			v = self.tree_policy()
			reward = v.default_policy()
			v.backup(reward)
			n -= 1
		best, _  = self.best_child()
		return best

