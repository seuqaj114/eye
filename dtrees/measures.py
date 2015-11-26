import numpy as np

def custom_entropy(patches, offsets, steepness=0.125):

    assert type(offsets) == type(np.array(0)), "'offsets' must be a numpy array."

    size = len(patches)
    offset_norms = np.linalg.norm(offsets,axis=2)
    patch_kp_probs = np.exp(-offset_norms/steepness)

    patch_prob_sum = np.sum(patch_kp_probs,axis=0)

    kp_terms = (patch_prob_sum/size)*np.log(patch_prob_sum/size)

    entropy = -np.sum(kp_terms)

    return entropy
    
def info_gain(params, entropy, **kwargs):
    pass    