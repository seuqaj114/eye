import numpy as np

def binary_test(channels, params, th=0):

    def patch_sum(R, channel):
        area = (R[3] - R[1])*(R[2] - R[0])
        return np.sum(channel[R[1]:R[3],R[0]:R[2]])/float(area)

    assert "R1" in params and "R2" in params and "a" in params, "'params' must contain R1, R2 and a."

    #   R1 and R2 must have the form
    #   [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    R1 = params["R1"]
    R2 = params["R2"]
    a  = params["a"]

    return th < (patch_sum(R1, channels[a]) - patch_sum(R2, channels[a]))
