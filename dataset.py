import os
import numpy as np
import cv2
from math import ceil
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

        self.images     = []
        self.eyes       = []
        self.keypoints  = []
        self.faces      = []
        self.faces_bbox = []

        self.IMG_PATH = os.path.abspath("assets/images")
        self.EYE_PATH = os.path.abspath("assets/eyepos")
        self.KP_PATH  = os.path.abspath("assets/points_20")

        for index in self.index_list:
            #   Store eye positions
            eye_filename = os.path.join(self.EYE_PATH,"BioID_"+str(index).zfill(4)+".eye")
            eyes = self.format_eye_file(eye_filename)
            self.eyes.append(eyes)

            #   Store images
            image_filename = os.path.join(self.IMG_PATH,"BioID_"+str(index).zfill(4)+".pgm")
            image = cv2.imread(image_filename,0)
            self.images.append(image)

            #   Store keypoints
            keypoints_filename = os.path.join(self.KP_PATH,"bioid_"+str(index).zfill(4)+".pts")
            keypoints = self.format_keypoints_file(keypoints_filename)
            self.keypoints.append(keypoints)

            face_bbox = self.extract_face_bbox(image, keypoints)
            self.faces_bbox.append(face_bbox)


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

    def extract_face_bbox(self, image, keypoints):
        margin = 10

        x_locations = [kp[0] for kp in keypoints]
        y_locations = [kp[1] for kp in keypoints]

        face_bbox = [max(0,min(x_locations)-margin), max(0,min(y_locations)-margin*4), min(image.shape[1],max(x_locations)+margin), min(image.shape[0],max(y_locations)+margin)]

        diff = (face_bbox[2] - face_bbox[0]) - (face_bbox[3] - face_bbox[1])
        #   If box is higher than wider, increase width
        if diff > 0:
            #   We have to make sure the boxes don't go out of the image
            face_bbox[1] = max(0,face_bbox[1]-abs(diff)/2)
            face_bbox[3] = min(image.shape[0]-1,face_bbox[3]+int(ceil(abs(diff)/2.0)))

            #   If it isn't possible to fully increase width, decrease height
            diff2 = (face_bbox[2] - face_bbox[0]) - (face_bbox[3] - face_bbox[1])
            if diff2 > 0:
                face_bbox[0] += abs(diff2)/2
                face_bbox[2] -= int(ceil(abs(diff2)/2.0))

        #   Otherwise, increase height
        else:
            #   We have to make sure the boxes don't go out of the image
            face_bbox[0] = max(0,face_bbox[0]-abs(diff)/2)
            face_bbox[2] = min(image.shape[1]-1,face_bbox[2]+int(ceil(abs(diff)/2.0)))

            #   If it isn't possible to fully increase height, decrease width
            diff2 = (face_bbox[2] - face_bbox[0]) - (face_bbox[3] - face_bbox[1])
            if diff2 < 0:
                face_bbox[1] += abs(diff2)/2
                face_bbox[3] -= int(ceil(abs(diff2)/2.0))

        assert (face_bbox[2] - face_bbox[0]) - (face_bbox[3] - face_bbox[1]) == 0, "Bounding box must be a square."

        return face_bbox

    def __iter__(self):
        return self

    def next(self):
        try:
            self.__curr_index += 1
            return (self.images[self.__curr_index-1], self.eyes[self.__curr_index-1], self.keypoints[self.__curr_index-1])
        except IndexError:
            self.__curr_index = 0
            raise StopIteration