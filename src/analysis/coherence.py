import csv
import sys
import itertools
import math
import rpy2
import rpy2.robjects as ro
import numpy as np
from src.analysis.analysis_utils import get_result_file_dict
from src.analysis.scrape_for_go_id import build_map
from tqdm import tqdm


def setup_for_r():
    ro.r('''
            library(GO.db)
            library(org.Hs.eg.db)
            library(GOSemSim)
            d <- godata("org.Hs.eg.db", ont="BP", computeIC=TRUE)
    ''')

def r_go_sim(goid1, goid2):
    go_sim_call = 'goSim("{0}", "{1}", d, measure="Lin")'.format(goid1, goid2)
    result = ro.r(go_sim_call)
    return result[0]

def generate_pairs_map(unit_geneset_map, geneset_id_map):
    pairs_map = {}
    for unit in tqdm(unit_geneset_map):
        genesets = unit_geneset_map[unit]
        pairs_map[unit] = []
        for geneset1, geneset2 in itertools.combinations(genesets, 2):
            if geneset1 in geneset_id_map and geneset2 in geneset_id_map:
                pairs_map[unit].append((geneset_id_map[geneset1],
                                        geneset_id_map[geneset2]))
    return pairs_map

def compute_mean_coherence(unit_geneset_pairs_map):
    coherence = {}
    for unit in tqdm(unit_geneset_pairs_map):
        coherences = []
        for pair in unit_geneset_pairs_map[unit]:
            try:
                coher = r_go_sim(pair[0], pair[1])
                coherences.append(coher)
            except rpy2.rinterface.RRuntimeError:
                pass

        coherence[unit] = np.mean(coherences)

    return coherence


def main(path):
    biology_filename = path+"biology.txt"

    # This file contains a list of all go terms
    go_terms_filename = path+"go_terms.txt"

    # This file has term -> goID map
    term_map_filename = path+"go_term_map.txt"

    coherence, unit_geneset_map = get_coherence(biology_filename, go_terms_filename, term_map_filename, use_web=True)

    # Write coherence to file
    coherence_list = []

    with open(path+"coherence.csv", 'w') as file:
        table = csv.writer(file)

        for unit in coherence:
            if not math.isnan(coherence[unit]):
                print(unit_geneset_map[unit])
                print('Coherence {}'.format(coherence[unit]))
                coherence_list.append(coherence[unit])
                table.writerow(unit, (coherence[unit], len(unit_geneset_map[unit]), unit_geneset_map[unit]))

    print('Coherence mean:{}'.format(np.mean(coherence_list)))
    print('Coherence stdv:{}'.format(np.std(coherence_list)))
    print('Coherence totl:{}'.format(len(coherence_list)))


def get_coherence(biology_filename, go_terms_filename, term_map_filename, use_web=False):
    term_id_map = build_map(go_terms_filename, term_map_filename, use_web)

    unit_geneset_map = get_result_file_dict(biology_filename)

    unit_geneset_map = remove_non_GO_terms(unit_geneset_map)

    pairs_map = generate_pairs_map(unit_geneset_map, term_id_map)
    setup_for_r()
    coherence = compute_mean_coherence(pairs_map)
    return coherence, unit_geneset_map


def remove_non_GO_terms(unit_geneset_map):
    for unit in unit_geneset_map:
        geneset_list = []
        for geneset in unit_geneset_map[unit]:
            if 'GO_' in geneset:
                geneset_list.append(geneset)
        unit_geneset_map[unit] = geneset_list
    return unit_geneset_map


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "C:\\Users\\pulin\\Projects\\self-taught\\results\\full_data_try4_best\\"
        path = "/mnt/c/Users/pulin/Projects/self-taught/results/full_data_try4_best/"

    main(path)

