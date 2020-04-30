"""
Helper functions to load the MNIST database of handwritten digits.

To run these functions you need the dataset in Matlab format from
Sam Roweis, available at (about 13 MB large):
http://www.cs.nyu.edu/~roweis/data/mnist_all.mat

The original data can be found here at http://yann.lecun.com/exdb/mnist/.

The size of the training sets varies between 5421 and 6742 samples, the
testing set from 958 to 1135.
"""

import numpy as np
import scipy.io

N_IDS = 10
FILENAME = "mnist_all.mat"

def _split(data, chunk_size):
    """Split data into arrays with length up to chunk_size and return as list.
    
    Unlike the numpy 'split' function this also works when the data length is
    not a multiple of chunk_size (the last chunk will simply be shorter).
    """
    n_chunks = int(np.ceil(len(data) / float(chunk_size)))
    split_data = [data[i_chunk*chunk_size :(i_chunk+1)*chunk_size]
                  for i_chunk in range(n_chunks)]
    return split_data

def get_data(prefix="train", max_chunk_size=6000):
    """Return two lists, one with the data chunks and one with the ids."""
    mat_data = scipy.io.loadmat(FILENAME)
    n_ids = 10
    data = []
    ids = []
    for id_num in range(n_ids):
        id_key = prefix + "%d" % id_num
        new_chunks = _split(mat_data[id_key].astype("float64"), max_chunk_size)
        data += new_chunks
        ids += [id_num] * len(new_chunks)
    return data, ids

    