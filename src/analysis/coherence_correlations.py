import sys
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
from src.analysis.coherence import get_coherence

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

    biology_filename = path+'best_sparse_biology.txt'

    coherence, unit_geneset_map = get_coherence(biology_filename, go_terms_filename, term_map_filename)

    weights_data = get_weights_data(path)

    unit_corr_matrix = get_corr_matrix(coherence, weights_data)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(unit_corr_matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.show()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "C:\\Users\\pulin\\Projects\\self-taught\\results\\full_data_try4_best\\"
        path = "/mnt/c/Users/pulin/Projects/self-taught/results/full_data_try4_best/"

    main(path)
