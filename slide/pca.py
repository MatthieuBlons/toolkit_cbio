from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm
import argparse
from glob import glob
from joblib import dump
import os
from slide.utils import read_h5_features


def check_batch(batch):
    """Checks if batch is big enough for the incremental PCA to be
    efficient.

    Parameters
    ----------
    batch : list
        list of matrix, each matrix corresponding to a WSI divided in $row tiles

    Returns
    -------
    bool
        Is the batch big enough ?
    """
    if batch:
        n_tiles = np.sum([x.shape[0] for x in batch])
        n_features = batch[-1].shape[1]
        ans = n_tiles >= n_features
    else:
        ans = False
    return ans

def get_feat_files(feat_dir):
    files = glob(os.path.join(feat_dir,'*.h5'))
    return files


def fit(feat_dir):
    files = get_feat_files(feat_dir)
    ipca = IncrementalPCA()
    batch = []
    for file in tqdm(files):
        _, mat = read_h5_features(file)
        if len(mat.shape) == 1:
            mat = np.expand_dims(mat, 0)
        if mat.sum() == 0:
            continue
        if check_batch(batch):
            batch = np.vstack(batch)
            ipca.partial_fit(X=batch)
            batch = []
        else:
            batch.append(mat)
    return files, ipca
