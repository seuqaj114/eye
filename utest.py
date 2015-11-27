from dtrees import DecisionTree
from dtrees.features import binary_test, Patcher
from dtrees.measures import custom_info_gain

from dataset import BioIDDataProvider

import numpy as np

def main():
    
    data = BioIDDataProvider(10)
    print data.keypoints[0]
    tree = DecisionTree(custom_info_gain, binary_test)

    patcher = Patcher()
    
    X = np.array([])
    Y = np.array([])


    for (image,eyes,keypoints) in data:
        patches = patcher.generate_patches(image, 5)
        offsets = patcher.calculate_offsets_from_patches(patches, keypoints)

        #   Finish this part


    #tree.split_node(tree.nodes[0], X_subset, Y_subset, attributes_list)



if __name__ == "__main__":
    main()