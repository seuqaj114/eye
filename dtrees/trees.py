import numpy as np

from features import Patcher

class DecisionTree():
    def __init__(self, criterion, split_func):
        self.criterion    = criterion
        self.split_func   = split_func
        self.nodes        = [Node()]

    def load_data_provider(self, data):
        self.data = data

        patcher = Patcher()
        
        self.patches = np.array([])
        self.offsets = np.array([])
        self.indexes = []

        #   Extract patches and offsets from data provided.
        #   TODO: Restrict patch extraction to face bounding box

        for i, (image,eyes,keypoints) in enumerate(self.data):
            patches = patcher.generate_patches(image, 5)
            offsets = patcher.calculate_offsets_from_patches(patches, keypoints)
            indexes = [i]*patches.shape[0]

            if not self.patches.size:
                self.patches = patches
                self.offsets = offsets
                self.indexes = indexes
            else:
                self.patches = np.concatenate((self.patches, patches),axis=0)
                self.offsets = np.concatenate((self.offsets, offsets),axis=0)
                self.indexes.extend(indexes)

        self.params_list = patcher.generate_parameters()

    def train(self, cross_val=True):
        assert hasattr(self, "patches") and hasattr(self, "offsets") and hasattr(self, "indexes"), "Data not loaded."

    def apply(self, X):
        raise NotImplementedError()

    def split_node(self, node, patches_subset, offsets_subset, indexes_subset, params_list):
        
        criterion_best = 0
        criterion_indexes = []

        for params in params_list:
            print "Params: {}".format(params)

            split_indexes = np.array([])

            for i in range(patches_subset.shape[0]):
                split_indexes = np.append(split_indexes, self.split_func(patches_subset[i], self.data.images[indexes_subset[i]], params))

            #print "Split indexes: {}".format(split_indexes)
            criterion_result = self.criterion(patches_subset, offsets_subset, split_indexes)
            print "Info-gain: {}".format(criterion_result)

            if criterion_result > criterion_best:
                criterion_best = criterion_result
                criterion_indexes = split_indexes

            print "\n"

        print "Best info-gain: {}".format(criterion_best)
        print "Best split indexes: {}".format(criterion_indexes)

class Node():
    def __init__(self, parent=None, children=[]):
        self.parent   = parent
        self.children = children
        self.is_leaf  = False
        self.prob_dist = None

    def add_children(self, children):
        if type(children) != type(list()):
            children = [children]

        self.children.extend(children)

    def add_parent(self, parent):
        self.parent = parent
