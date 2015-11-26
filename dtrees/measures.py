import numpy as np

def custom_entropy(patches, offsets, steepness=0.125):

    assert type(offsets) == type(np.array(0)), "'offsets' must be a numpy array."

    num_patches = len(patches)
    offset_norms = np.linalg.norm(offsets,axis=2)
    patch_kp_probs = np.exp(-offset_norms/steepness)
    normalization = np.sum(patch_kp_probs,axis=1)
    patch_kp_probs = patch_kp_probs/normalization[:,None]

    patch_prob_sum = np.sum(patch_kp_probs,axis=0)

    kp_terms = (patch_prob_sum/num_patches)*np.log(patch_prob_sum/size)

    entropy = -np.sum(kp_terms)

    return entropy
    
def custom_info_gain(patches, offsets, split_indexes):
    pass