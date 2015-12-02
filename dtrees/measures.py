import numpy as np

def custom_entropy(patches, offsets, steepness=20.0):

    if type(offsets) != type(np.array(0)):
        offsets = np.array(offsets)

    if patches.shape[0] == 0:
        return 0

    num_patches = len(patches)
    offset_norms = np.linalg.norm(offsets,axis=2)
    patch_kp_probs = np.exp(-offset_norms/steepness)
    normalization = np.sum(patch_kp_probs,axis=1)
    patch_kp_probs = patch_kp_probs/normalization[:,None]

    patch_prob_sum = np.sum(patch_kp_probs,axis=0)

    kp_terms = (patch_prob_sum/num_patches)*np.log(patch_prob_sum/num_patches)

    entropy = -np.sum(kp_terms)

    return entropy
    
def custom_info_gain(patches, offsets, split_indexes):
    
    assert type(split_indexes) == type(np.array(0)), "'split_indexes' must be a numpy array."

    right_patches = patches[split_indexes == 1]
    right_offsets = offsets[split_indexes == 1]

    left_patches = patches[split_indexes == 0]
    left_offsets = offsets[split_indexes == 0]

    info_gain = custom_entropy(patches, offsets) - ( \
                (right_patches.shape[0])*custom_entropy(right_patches, right_offsets) + \
                (left_patches.shape[0])*custom_entropy(left_patches, left_offsets) \
                )/patches.shape[0]

    return info_gain


    