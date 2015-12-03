import numpy as np
import sys

from features import Patcher
from utils import timeit

class DecisionTree():
    def __init__(self, criterion, split_func, max_depth):
        self.criterion    = criterion
        self.split_func   = split_func
        self.nodes        = []
        self.max_depth    = max_depth

    def load_data_provider(self, data, patch_size=20, num_patches=50, num_params=20):
        self.data = data

        patcher = Patcher(patch_size)
        
        self.patches = np.array([])
        self.offsets = np.array([])
        self.indexes = np.array([])

        #   Extract patches and offsets from data provided.
        #   TODO: Restrict patch extraction to face bounding box

        for i, (image, eyes, keypoints, face, face_bbox, rkeypoints) in enumerate(self.data):
            patches = patcher.generate_patches(face, num_patches)
            offsets = patcher.calculate_offsets_from_patches(patches, rkeypoints)
            indexes = np.ones(patches.shape[0],dtype="int")*i

            if not self.patches.size:
                self.patches = patches
                self.offsets = offsets
                self.indexes = indexes
            else:
                self.patches = np.concatenate((self.patches, patches),axis=0)
                self.offsets = np.concatenate((self.offsets, offsets),axis=0)
                self.indexes = np.concatenate((self.indexes, indexes),axis=0)

        self.params_list = patcher.generate_parameters(num_params)

    @timeit
    def train(self, cross_val=True):
        #   Note: the performance is highly suboptimal right now, because
        #   the spliting criterion is performed at every node.
        #   Ideally we should calculate them all at the beginning and
        #   just access them as we create the tree.
        
        assert hasattr(self, "patches") and hasattr(self, "offsets") and hasattr(self, "indexes"), "Data not loaded."

        node = Node(level=1)
        self.nodes.append(node)

        self.split_node(node, self.patches, self.offsets, self.indexes, self.params_list)

    def apply(self, X):
        raise NotImplementedError()

    def split_node(self, node, patches, offsets, indexes, params_list):
        
        if node.level == self.max_depth:
            node.is_leaf = True
            print "Leaf reached: level {}".format(node.level)
            print "Indexes: {}".format(indexes)

            node.prob_dist = self.create_gaussian_parameters(offsets)
            return

        criterion_best = 0.0
        best_split_indexes = []
        best_params_index = None

        for params_index in range(params_list.shape[0]):
            #print "Params: {}".format(params_list[params_index])

            split_indexes = np.array([])

            for i in range(patches.shape[0]):
                sys.stdout.write("\rPatch: {}/{} , Params: {}/{}".format(i,patches.shape[0],params_index,params_list.shape[0]))
                sys.stdout.flush() 
                split_indexes = np.append(split_indexes, self.split_func(patches[i], self.data.faces[indexes[i]], params_list[params_index]))

            #print "Split indexes: {}".format(split_indexes)
            criterion_result = self.criterion(patches, offsets, split_indexes)
            #print "Info-gain: {}".format(criterion_result)

            if criterion_result > criterion_best:
                criterion_best = criterion_result
                best_split_indexes = split_indexes
                best_params_index = params_index

            #print "\n"

        print "\nBest info-gain: {}".format(criterion_best)
        #print "Best split indexes: {}".format(best_split_indexes)

        #   If node is pure, reached leaf node
        if not best_params_index:
            node.is_leaf = True
            print "Leaf reached: level {}".format(node.level)
            print "Indexes: {}".format(indexes)
            node.prob_dist = self.create_gaussian_parameters(offsets)
            #   TODO: create probability distributions   

        #   Otherwise, continue recursion
        else:
            print "Non-leaf reached."
            next_params_list = params_list[np.arange(params_list.shape[0])!=best_params_index]

            node.params = params_list[best_params_index]

            left_patches = patches[best_split_indexes == 0]
            left_offsets = offsets[best_split_indexes == 0]
            left_indexes = indexes[best_split_indexes == 0]

            right_patches = patches[best_split_indexes == 1]
            right_offsets = offsets[best_split_indexes == 1]
            right_indexes = indexes[best_split_indexes == 1]

            left_node = Node(level=node.level+1)
            node.children.append(left_node)
            self.nodes.append(left_node)
            self.split_node(left_node, left_patches, left_offsets, left_indexes, next_params_list)

            right_node = Node(level=node.level+1)
            node.children.append(right_node)
            self.nodes.append(right_node)
            self.split_node(right_node, right_patches, right_offsets, right_indexes, next_params_list)

    def create_gaussian_parameters(self, offsets):
        mean = np.mean(offsets, axis=0)

        #   I had to brute-force this one because couldn't figure out how to do it all at once...
        aux = np.swapaxes(offsets, 0, 1)
        cov = []

        for i in range(aux.shape[0]):
            cov.append(np.dot((aux[i]-aux[i].mean(0)).T,aux[i]-aux[i].mean(0))/aux[i].shape[0])

        cov = np.array(cov)

        return mean, cov

class Node():
    def __init__(self, level=None, children=[]):
        self.children = children
        self.params = {}
        self.is_leaf  = False
        self.prob_dist = None
        self.level = level

    def add_children(self, children):
        if type(children) != type(list()):
            children = [children]

        self.children.extend(children)

