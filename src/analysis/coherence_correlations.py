import sys
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from src.analysis.coherence import get_coherence
from src.analysis.encoding import get_features

def get_weights_data(path):
    weights_filename = os.path.join(path, 'features_model.csv')
    weights_data = pd.read_csv(open(weights_filename, 'r'), sep='\t')
    #weights_data = weights_data.set_index('Gene')
    #weights_data = weights_data.T
    return weights_data


def get_corr_matrix(coherence, weights_data):
    units = []
    for unit in coherence:
        if not math.isnan(coherence[unit]):
            units.append(str(unit - 1))
    unit_matrix = weights_data[units]
    unit_corr_matrix = unit_matrix.corr()

    return unit_corr_matrix

def main(path):

    # This file contains a list of all go terms
    go_terms_filename = path+"go_terms.txt"

    # This file has term -> goID map
    term_map_filename = path+"go_term_map.txt"

    biology_filename = path+'biology.txt'

    coherence, unit_geneset_map = get_coherence(path)

    if os.path.exists(os.path.join(path, 'features_model.csv')):
        weights_data = get_weights_data(path)
    else:
        weights_data = get_features(path)

    unit_corr_matrix = get_corr_matrix(coherence, weights_data)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(unit_corr_matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.savefig(path+'coherence_corr.png')
    plt.show()


def plot_corr(df, size=20):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()

    # Plot the correlation matrix
    fig = plt.figure()

    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);

    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    plt.savefig(path+'coherence_corr_clustered.png')
    plt.show()

def plot_cluster_corr(df, size=20):
    cluster_th = 4

    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5 * d.max(), 'distance')

    columns = [df.columns.tolist()[i] for i in list(np.argsort(ind))]
    df = df.reindex_axis(columns, axis=1)

    unique, counts = np.unique(ind, return_counts=True)
    counts = dict(zip(unique, counts))

    i = 0
    j = 0
    columns = []
    for cluster_l1 in set(sorted(ind)):
        j += counts[cluster_l1]
        sub = df[df.columns.values[i:j]]
        if counts[cluster_l1] > cluster_th:
            X = sub.corr().values
            d = sch.distance.pdist(X)
            L = sch.linkage(d, method='complete')
            ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
            col = [sub.columns.tolist()[i] for i in list((np.argsort(ind)))]
            sub = sub.reindex_axis(col, axis=1)
        cols = sub.columns.tolist()
        columns.extend(cols)
        i = j
    df = df.reindex_axis(columns, axis=1)

    plot_corr(df, size)
    return df

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "C:\\Users\\pulin\\Projects\\self-taught\\results\\full_data_try4_best\\"
        path = "/mnt/c/Users/pulin/Projects/self-taught/results/full_data_try4_best2/"

    main(path)
