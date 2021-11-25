import numpy as np
from numpy.random.mtrand import shuffle


def train_test_split(x, y, test_ration=0.2, seed=None):
    assert x.shape[0] == y.shape[0], 'the size x must be equal to the size of y'
    assert 0.0 <= test_ration <= 1, 'test_ration must be valid'
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(x))
    indexcount = int(test_ration*len(x))
    tests = shuffle_indexes[:indexcount]
    trians = shuffle_indexes[indexcount:]
    return x[trians], y[trians], x[tests], y[tests]
