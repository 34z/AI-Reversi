from AIPlayer import *
from mcts import *
from copy import deepcopy


class ReversiNode(MCTreeNode):
    def __init__(self, parent=None, board=None, color=None, from_action=None):
        super(ReversiNode, self).__init__(parent)
        if parent is None:
            self.board = board
            self.color = color
            self.root_color = color
        else:
            self.board = self.parent.board
            self.from_action = from_action
            self.root_color = self.parent.root_color
            if self.from_action is not None:
                self.color = reverse_color(self.parent.color)
                self.flipped = self.board._move(self.from_action, self.parent.color)
                self.backup_func = lambda : self.board.backpropagation(self.from_action, self.flipped, self.parent.color)
            else:
                self.color = self.parent.color
        self.actions = list(self.board.get_legal_actions(self.color))
        random.shuffle(self.actions)
        actions_t = list(self.board.get_legal_actions(reverse_color(self.color)))
        self.is_over = len(self.actions) == 0 and len(actions_t) == 0
        self.select_func = lambda child: self.board._move(child.from_action, self.color)
    
    @property
    def is_leaf(self):
        return self.is_over
    
    def is_fully_expanded(self):
        return self.n_ch == len(self.actions) and len(self.actions) > 0
    
    def expand(self):
        assert not self.is_fully_expanded()
        action = None
        if len(self.actions) > 0:
            action = self.actions[self.n_ch]
        child = ReversiNode(parent=self, from_action=action)
        self.children.append(child)
        return child

    def default_policy(self):
        board = deepcopy(self.board)
        p1 = AIPlayer(self.color)
        p2 = AIPlayer(reverse_color(self.color))
        p1.search, p2.search = 'random', 'random'
        while True:
            over = True
            action = p1.get_move(board)
            if action:
                board._move(action, p1.color)
                over = False
            action = p2.get_move(board)
            if action:
                board._move(action, p2.color)
                over = False
            if over:
                break
        winner, _ = board.get_winner()
        # board.display()
        reward = 0
        if match_color(self.root_color, winner):
            reward = 1
        return reward 
        