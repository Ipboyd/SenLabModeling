import numpy as np


def pinit(batch_size, num_params):
    rng = np.random
    p = np.zeros((num_params,batch_size))
    p[0:4,:] = rng.uniform(0.0, 0.08, size=(4, batch_size)).astype(np.float32) #GSYNs
    return p
