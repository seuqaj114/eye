
class DecisionTree():
    def __init__(self, criterion, split_func):
        self.criterion    = criterion
        self.split_func   = split_func
        self.nodes        = [Node()]

    def train(self, X, Y, cross_val=True):
        raise NotImplementedError()

    def apply(self, X):
        raise NotImplementedError()

    def split_node(self, node, X_subset, Y_subset, attributes_list):

        #   'information_gains' will have the same length as 'attributes_list'
        criterion_best = 0
        criterion_indexes = []

        for attributes in attributes_list:
            possible_split_indexes = map(lambda x: self.split_func(x,attributes), X_subset)
            criterion_result = self.criterion(X, Y, possible_split_indexes)

            if criterion_result > criterion_best:
                criterion_best = criterion_result
                criterion_indexes = possible_split_indexes


class Node():
    def __init__(self, parent=None, children=[]):
        self.parent   = parent
        self.children = children
        self.is_leaf  = False

    def add_children(self, children):
        if type(children) != type(list()):
            children = [children]

        self.children.extend(children)

    def add_parent(self, parent):
        self.parent = parent
