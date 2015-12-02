import os
import numpy as np
import cv2
from random import shuffle

MAX_NUM_IMAGES = 1521

class BioIDDataProvider():
    
    def __init__(self, num_images=MAX_NUM_IMAGES, randomize=False):

        self.num_images = max(1,min(num_images, MAX_NUM_IMAGES))
        
        if randomize:
            self.index_list = range(MAX_NUM_IMAGES)
            shuffle(self.index_list)
            self.index_list = self.index_list[:self.num_images]
        else:
            self.index_list = range(self.num_images)

        self.images    = []
        self.eyes      = []
        self.keypoints = []

        self.IMG_PATH = os.path.abspath("assets/images")
        self.EYE_PATH = os.path.abspath("assets/eyepos")
        self.KP_PATH  = os.path.abspath("assets/points_20")

        for index in self.index_list:
            #   Store eye positions
            eye_filename = os.path.join(self.EYE_PATH,"BioID_"+str(index).zfill(4)+".eye")
            self.eyes.append(self.format_eye_file(eye_filename))

            #   Store images
            image_filename = os.path.join(self.IMG_PATH,"BioID_"+str(index).zfill(4)+".pgm")
            self.images.append(cv2.imread(image_filename,0))

            #   Store keypoints
            keypoints_filename = os.path.join(self.KP_PATH,"bioid_"+str(index).zfill(4)+".pts")
            self.keypoints.append(self.format_keypoints_file(keypoints_filename))

        self.__curr_index = 0

    def format_eye_file(self,filename):
        with open(filename,"r") as fp:
            fp.next()

            #   The coordinates are in the 2nd line of the file
            coords = map(int,fp.next().strip().split())

        return coords

    def format_keypoints_file(self,filename):
        
        #   See http://personalpages.manchester.ac.uk/staff/timothy.f.cootes/data/bioid_points.html for format
        with open(filename,"r") as fp:
            coords = []
            for line in fp.readlines()[3:-1]:
                coords.append(map(lambda x: int(float(x)),line.strip().split()))

        return coords

    def __iter__(self):
        return self

    def next(self):
        try:
            self.__curr_index += 1
            return (self.images[self.__curr_index-1], self.eyes[self.__curr_index-1], self.keypoints[self.__curr_index-1])
        except IndexError:
            self.__curr_index = 0
            raise StopIteration

            
