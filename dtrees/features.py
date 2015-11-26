import numpy as np

def binary_test(R1, R2, a, channels):

    def patch_sum(R, channel):
        area = (R[3] - R[1])*(R[2] - R[0])
        return np.sum(channel[R[1]:R[3],R[0]:R[2]])/float(area)

    #   R1 and R2 must have the form
    #   [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    return patch_sum(R1, channels[a]) - patch_sum(R2, channels[a])
