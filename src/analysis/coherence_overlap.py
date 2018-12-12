import sys
import os
import trio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from src.analysis.coherence import get_coherence
from src.analysis.encoding import get_features

def get_weights_data(path):
    weights_data = None

    async def print_wait():
        while True:
            await trio.sleep(5)
            print('.', end='')

    async def nursery_get(path, nursery):
        nonlocal weights_data
        weights_filename = os.path.join(path, 'features_model.csv')
        weights_data = pd.read_csv(open(weights_filename, 'r'), sep='\t')
        nursery.cancel_scope.cancel()

    async def async_get(path):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(print_wait)
            nursery.start_soon(nursery_get, path, nursery)

    trio.run(async_get, path)

    #weights_data = weights_data.set_index('Gene')
    #weights_data = weights_data.T
    return weights_data

def overlap_matrix(sets):
    K = len(sets)
    overlap = np.empty((K, K), dtype=float)
    collect_overlap = []
    for i, key_i in enumerate(sets):
        for j, key_j in enumerate(sets):
            if i > j:
                overlap_value = len(sets[key_i].intersection(sets[key_j]))
                overlap[i, j] = overlap_value
                overlap[j, i] = overlap_value
                collect_overlap.append(overlap_value)

    return pd.DataFrame(overlap, index=list(sets), columns=list(sets)), np.mean(collect_overlap), np.std(collect_overlap)

def factors(coherence):
    return sorted(set(coherence.values()), reverse=True)

def greater_than(value):
    return lambda x: x>value

def less_than(value):
    return lambda x: x<value

def equals(value):
    return lambda x: x == value

def in_factor(factors):
    return lambda x: x in factors

def filter(filters, dictionary):
    return { k: v for k,v in dictionary.items() if all(f(v) for f in filters) }

def filter_old(coherence, factor=None, value=None, greater_than=0.0):
    filtered = []
    if greater_than > 0.0:
        greater=True
    else:
        greater=False
    if factor is not None:
        value=sorted(set(coherence.values()), reverse=True)[factor-1]
    for unit in coherence:
        if greater:
            if coherence[unit]>greater_than:
                filtered.append(unit)
        else:
            if coherence[unit] == value:
                filtered.append(unit)
    return filtered

def get_top_genes(weights_data, filtered_units, top_x_pct_weights=0.1):
    top_genes_map = {}
    # Get absolute value of all weights for each unit
    num_of_genes = int(top_x_pct_weights*len(weights_data['0']))
    for unit in filtered_units:
        top_genes_map[unit]=set(weights_data[str(unit-1)].abs().sort_values(ascending=False)[:num_of_genes].index)

    return top_genes_map


def main(path):

    # This file contains a list of all go terms
    go_terms_filename = path+"go_terms.txt"

    # This file has term -> goID map
    term_map_filename = path+"go_term_map.txt"

    biology_filename = path+'biology.txt'

    coherence, unit_geneset_map = get_coherence(path)

    coherence_factors = factors(coherence)
    relevant_factors = [coherence_factors[i] for i in (0,1,2,3,4)]

    filters = [ less_than(1.0)
               ]

    filtered = filter(filters, coherence)

    sorted_filtered_units = list(sorted(filtered, key=filtered.get, reverse=True))

    if os.path.exists(os.path.join(path, 'features_model.csv')):
        weights_data = get_weights_data(path)
    else:
        weights_data = get_features(path)

    top_genes_map = get_top_genes(weights_data, sorted_filtered_units)

    # pairwise compare overlap of top x pct weight genes
    overlap_correlation, mean, std = overlap_matrix(top_genes_map)
    print('Mean of matrix: ', mean)
    print('Stdv of matrix: ', std)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(overlap_correlation)
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

def plot_cluster_corr(df, size=20, corr=False):
    cluster_th = 4

    if corr:
        X = df.values
    else:
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

    if corr:
        fig = plt.figure()

        ax = fig.add_subplot(111)
        cax = ax.matshow(df)
        fig.colorbar(cax)
        plt.savefig(path + 'coherence_corr.png')
        plt.show()
    else:
        plot_corr(df, size)
    return df

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "C:\\Users\\pulin\\Projects\\self-taught\\results\\full_data_try4_best\\"
        path = "/mnt/c/Users/pulin/Projects/self-taught/results/full_data_try4_best2/"

    main(path)
