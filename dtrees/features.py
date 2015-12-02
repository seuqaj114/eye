import numpy as np
from random import randint
from numpy import typecodes

class Patcher():
    def __init__(self, size=32):
        self.size = size

    #   This is for a single image
    def generate_patches(self, image, n):
        #   Avoid negative values
        n = max(1,n)

        #   Each patch has the form
        #   [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]
        patches = []
        for i in range(n):
            
            #   Generate top left corner
            y = randint(0, image.shape[0]-1-self.size)
            x = randint(0, image.shape[1]-1-self.size)
            
            patches.append([[x, y], [x+self.size, y+self.size]])

        return np.array(patches)

    #   This is for a single image
    def calculate_offsets_from_patches(self, patches, keypoints):

        if type(patches) != type(np.array(0)):
            patches = np.array(patches)
        if type(keypoints) != type(np.array(0)):
            keypoints = np.array(keypoints)

        centroids = centroid(patches)
        offsets = (np.repeat(centroids,keypoints.shape[0],axis=0)-np.tile(keypoints,(centroids.shape[0],1))).reshape(centroids.shape[0],keypoints.shape[0],2)

        return offsets

    def generate_parameters(self, n=10):
        
        params_list = []

        for i in range(n):

            params = {}

            for j in range(1,3):
                #   Generate top left corner
                top_left_y = randint(0, 3*(self.size-1)/4)
                top_left_x = randint(0, 3*(self.size-1)/4)

                #   Generate bottom right corner
                #   Parameter regions must be at least 25 pixels in area
                bottom_right_y = randint(top_left_y+5, self.size-1)
                bottom_right_x = randint(top_left_x+5, self.size-1)

                params["R"+str(j)] = np.array([[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]])

            params_list.append(params)  

        return params_list

def binary_test(patch, channels, params, th=0):
    """ This function should probably be vectorized... """
    """ TODO: Add channel parameter 'a' """

    def patch_sum(R):
        area = np.prod(R[1]-R[0])
        return np.sum(channels[patch[0][1]+R[0][1]:patch[1][1]+R[1][1],patch[0][0]+R[0][0]:patch[1][0]+R[1][0]])/float(area)

    assert "R1" in params and "R2" in params, "'params' must contain R1, R2."

    #   R1 and R2 must have the form
    #   [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]

    R1 = params["R1"]
    R2 = params["R2"]
    #a  = params["a"]

    return th < (patch_sum(R1) - patch_sum(R2))

def centroid(rect):
    return rect[...,0,...] + (rect[...,1,...] - rect[...,0,...])/2