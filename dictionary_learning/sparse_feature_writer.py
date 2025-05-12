import h5py
from scipy.sparse import csr_matrix, vstack
import numpy as np
import os

# Uses Hierarchical Data Format version 5 (HDF5) to
# store the Compressed Sparse Row matrix (CSR) matrix in parts
class SparseFeatureWriter:
    def __init__(self, path):
        self.path = path
        self.total_rows = 0
        if not os.path.exists(path):
            with h5py.File(path, 'w') as f:
                # CSR matrix
                f.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
                f.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32') # column indices = feature idx
                f.create_dataset('indptr', shape=(1,), maxshape=(None,), dtype='int32')  # row pointers. starts at 0
                
                # Tokens
                f.create_dataset('tokens', shape=(0,), maxshape=(None,), dtype='int32')
                
                #Metadata
                f.attrs['shape'] = (-1, -1)  # updated later
        else:
            print("h5 file already exists. New data will be appended to previous data.")


    def append(self, sparse_batch: csr_matrix, token_batch: list[int]):
        with h5py.File(self.path, 'a') as f:
            batch_tokens = [tok for tok in token_batch]  # flatten tokens

            # Flatten CSR matrix to match tokens
            n_rows, n_cols = sparse_batch.shape
            assert len(batch_tokens) == n_rows, "Token count must match number of rows"

            # Current size
            curr_data = f['data'].shape[0]
            curr_indptr = f['indptr'].shape[0] - 1
            curr_tokens = f['tokens'].shape[0]

            # Resize
            f['data'].resize((curr_data + sparse_batch.data.shape[0],))
            f['indices'].resize((curr_data + sparse_batch.indices.shape[0],))
            f['indptr'].resize((curr_indptr + sparse_batch.indptr.shape[0],))
            f['tokens'].resize((curr_tokens + n_rows,))

            # Write data
            f['data'][curr_data:] = sparse_batch.data
            f['indices'][curr_data:] = sparse_batch.indices
            f['indptr'][curr_indptr + 1:] = sparse_batch.indptr[1:] + f['indptr'][curr_indptr]
            f['tokens'][curr_tokens:] = batch_tokens

            # Update shape attribute
            if f.attrs['shape'][1] == -1:
                f.attrs['shape'] = (0, n_cols)
            f.attrs['shape'] = (
                f.attrs['shape'][0] + n_rows,
                f.attrs['shape'][1]
            )