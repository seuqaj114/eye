

class DecisionTree():
    def __init__(self, criterion):
        self.criterion = criterion
        self.nodes     = []

    def train(self,X,Y, cross_val=True):
        raise NotImplementedError()

    def apply(self, X):
        raise NotImplementedError()


class Node():
    def __init__(self, parent=None, children=[]):
        self.parent   = parent
        self.children = children
        self.is_leaf  = False

    def add_children(self,children):
        if type(children) != type(list()):
            children = [children]

        self.children.extend(children)

    def add_parent(self, parent):
        self.parent = parent
