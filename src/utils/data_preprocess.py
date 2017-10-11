'''
from
https://github.com/elegant-scipy/notebooks/blob/master/ch2.ipynb
'''
import numpy as np
import pandas as pd
import os
import pickle as pkl

from scipy import stats

def quantile_norm(X):
    """Normalize the columns of X to each have the same distribution.

    Given an expression matrix (microarray data, read counts, etc) of M genes
    by N samples, quantile normalization ensures all samples have the same
    spread of data (by construction).

    The data across each row are averaged to obtain an average column. Each
    column quantile is replaced with the corresponding quantile of the average
    column.

    Parameters
    ----------
    X : 2D array of float, shape (M, N)
        The input data, with M rows (genes/features) and N columns (samples).

    Returns
    -------
    Xn : 2D array of float, shape (M, N)
        The normalized data.
    """
    # compute the quantiles
    quantiles = np.mean(np.sort(X, axis=0), axis=1)

    # compute the column-wise ranks. Each observation is replaced with its
    # rank in that column: the smallest observation is replaced by 1, the
    # second-smallest by 2, ..., and the largest by M, the number of rows.
    ranks = np.apply_along_axis(stats.rankdata, 0, X)

    # convert ranks to integer indices from 0 to M-1
    rank_indices = ranks.astype(int) - 1

    # index the quantiles for each rank with the ranks matrix
    Xn = quantiles[rank_indices]

    return(Xn)

def quantile_norm_log(X):
    logX = np.log(X + 1)
    logXn = quantile_norm(logX)
    return logXn


def preprocess(geodb_frame):
    ''' sample as rows and features/genes as columns'''
    geodb_frame = geodb_frame.dropna(axis=0)
    

if __name__ == '__main__':

    pickled = False
    if not pickled:
        filename = os.path.join(os.path.pardir, os.path.pardir, 'data', 'final.txt')
        iter_csv = pd.read_csv(filename, sep='\t', index_col=0, chunksize=20000)
        print('read')
        df = pd.concat([chunk for chunk in iter_csv])
        print('loaded')
        df_mat = df.as_matrix()
        df_mat.shape()
        df_mat = quantile_norm(df_mat)
        print('normed')
        df.values = df_mat
        df.transpose()
        print(df.head())
        print(df.shape)
        pkl.dump(df, open('trnspd_q_normd_df.pkl'))


