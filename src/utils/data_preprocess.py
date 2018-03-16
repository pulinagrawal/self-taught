'''
from
https://github.com/elegant-scipy/notebooks/blob/master/ch2.ipynb
'''
import numpy as np
import pandas as pd
import os
import pickle as pkl
from src.utils import image as im


def rankdata(a, method='average'):
    """
    Assign ranks to data, dealing with ties appropriately.
    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.
    Parameters
    ----------
    a : array_like
        The array of values to be ranked.  The array is first flattened.
    method : str, optional
        The method used to assign ranks to tied elements.
        The options are 'average', 'min', 'max', 'dense' and 'ordinal'.
        'average':
            The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
        'min':
            The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
        'max':
            The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
        'dense':
            Like 'min', but the rank of the next highest element is assigned
            the rank immediately after those assigned to the tied elements.
        'ordinal':
            All values are given a distinct rank, corresponding to the order
            that the values occur in `a`.
        The default is 'average'.
    Returns
    -------
    ranks : ndarray
         An array of length equal to the size of `a`, containing rank
         scores.
    References
    ----------
    .. [1] "Ranking", http://en.wikipedia.org/wiki/Ranking
    Examples
    --------
    >>> from scipy.stats import rankdata
    >>> rankdata([0, 2, 3, 2])
    array([ 1. ,  2.5,  4. ,  2.5])
    >>> rankdata([0, 2, 3, 2], method='min')
    array([ 1,  2,  4,  2])
    >>> rankdata([0, 2, 3, 2], method='max')
    array([ 1,  3,  4,  3])
    >>> rankdata([0, 2, 3, 2], method='dense')
    array([ 1,  2,  3,  2])
    >>> rankdata([0, 2, 3, 2], method='ordinal')
    array([ 1,  2,  4,  3])
    """
    if method not in ('average', 'min', 'max', 'dense', 'ordinal'):
        raise ValueError('unknown method "{0}"'.format(method))

    arr = np.ravel(np.asarray(a))
    algo = 'mergesort' if method == 'ordinal' else 'quicksort'
    sorter = np.argsort(arr, kind=algo)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    if method == 'ordinal':
        return inv + 1

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    if method == 'dense':
        return dense

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    if method == 'max':
        return count[dense]

    if method == 'min':
        return count[dense - 1] + 1

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)

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
    ranks = np.apply_along_axis(rankdata, 0, X)

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

    pickled = True
    create_sets = True
    normed = False
    normed_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'transp_normd_1norm.pkl')
    print(os.path.realpath(normed_path))
    if not normed:
        if not pickled:
            filename = os.path.join('data', 'final.txt')
            iter_csv = pd.read_csv(filename, sep='\t', index_col=0, chunksize=20000)
            df = pd.concat([chunk for chunk in iter_csv])
        else:
            filename = os.path.join(os.path.pardir, os.path.pardir, 'data', 'final_transp_directpkl.pkl')
            df = pkl.load(open(filename, 'rb'))

        df = df.dropna(axis=0)
        df_mat = df.as_matrix()
        print(df_mat.shape)
        df_mat = quantile_norm(df_mat.transpose())
        '''
        df_mean = np.mean(df_mat, axis=1, keepdims=True)
        df_std = np.std(df_mat, axis=1, keepdims=True)
        df_mat = df_mat-df_mean
        df_mat = df_mat/df_std
        df_min = np.min(df_mat)
        df_mat = df_mat+abs(df_min)
        df_max = np.max(df_mat)
        df_mat = df_mat/df_max'''
        dfn = pd.DataFrame(df_mat.transpose(), index=df.index, columns=df.columns)
        dfn = dfn.dropna(axis=0)
        im.plot_genes(dfn.sample(1000))
        print(dfn.head())
        print(dfn.shape)
        pkl.dump(dfn, open(normed_path, 'wb'))


