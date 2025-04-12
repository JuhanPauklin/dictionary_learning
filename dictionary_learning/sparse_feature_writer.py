import h5py
from scipy.sparse import csr_matrix, vstack
import numpy as np
import os

class SparseFeatureWriter:
    def __init__(self, path):
        self.path = path
        self.total_rows = 0
        if not os.path.exists(path):
            with h5py.File(path, 'w') as f:
                f.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
                f.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
                f.create_dataset('indptr', shape=(1,), maxshape=(None,), dtype='int32')  # starts at 0
                f.attrs['shape'] = (-1, -1)  # updated later

    def append(self, sparse_batch: csr_matrix):
        with h5py.File(self.path, 'a') as f:
            n_rows, n_cols = sparse_batch.shape
            if f.attrs['shape'][1] == -1:
                f.attrs['shape'] = (0, n_cols)

            # Current shape info
            curr_data = f['data'].shape[0]
            curr_indptr = f['indptr'].shape[0] - 1  # indptr always has one more element

            # Resize datasets
            f['data'].resize((curr_data + sparse_batch.data.shape[0],))
            f['indices'].resize((curr_data + sparse_batch.indices.shape[0],))
            f['indptr'].resize((curr_indptr + sparse_batch.indptr.shape[0],))

            # Append data
            f['data'][curr_data:] = sparse_batch.data
            f['indices'][curr_data:] = sparse_batch.indices
            f['indptr'][curr_indptr + 1:] = sparse_batch.indptr[1:] + f['indptr'][curr_indptr]

            # Update shape metadata
            f.attrs['shape'] = (
                f.attrs['shape'][0] + n_rows,
                f.attrs['shape'][1]
            )