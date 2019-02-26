import csv
import os
import sys
import itertools
import math
import rpy2
import rpy2.robjects as ro
import numpy as np
from functools import lru_cache

from pylru import lrudecorator

sys.path.extend(["/mnt/c/Users/pulin/Projects/self-taught/src"])
from src.analysis.analysis_utils import get_result_file_dict
from src.analysis.scrape_for_go_id import build_map
from tqdm import tqdm

coherence_filename = "coherence.csv"

def setup_for_r():
    ro.r('''
            library(GO.db)
            library(org.Hs.eg.db)
            library(GOSemSim)
            d <- godata("org.Hs.eg.db", ont="BP", computeIC=TRUE)
    ''')

def r_go_sim(goid1, goid2):
    go_sim_call = 'goSim("{0}", "{1}", d, measure="Rel")'.format(goid1, goid2)
    result = ro.r(go_sim_call)
    return result[0]

def generate_pairs_map(unit_geneset_map, geneset_id_map):
    pairs_map = {}
    for unit in tqdm(unit_geneset_map):
        genesets = unit_geneset_map[unit]
        pairs_map[unit] = []
        if len(genesets)>1:
            for geneset1, geneset2 in itertools.combinations(genesets, 2):
                if geneset1 in geneset_id_map and geneset2 in geneset_id_map:
                    pairs_map[unit].append((geneset_id_map[geneset1],
                                            geneset_id_map[geneset2]))
        elif len(genesets)==1:
            pairs_map[unit]=[(genesets[0],)]
    return pairs_map

def compute_mean_coherence(unit_geneset_map, geneset_id_map):
    unit_geneset_pairs_map = generate_pairs_map(unit_geneset_map, geneset_id_map)
    flat_pairs={}
    for unit in tqdm(unit_geneset_pairs_map):
        for pair in unit_geneset_pairs_map[unit]:
            flat_pairs[pair]=0
    for pair in tqdm(flat_pairs):
        if len(pair)>1:
            try:
                coher = r_go_sim(pair[0], pair[1])
                flat_pairs[pair]=coher
            except rpy2.rinterface.RRuntimeError:
                pass
        else:
            flat_pairs[pair]=1.0

    coherence = {}
    for unit in tqdm(unit_geneset_pairs_map):
        coherences = []
        for pair in unit_geneset_pairs_map[unit]:
            coherences.append(flat_pairs[pair])

        coherence[unit] = np.mean(coherences)

    return coherence

'''
def compute_mean_coherence(unit_geneset_map, geneset_id_map):
    unit_geneset_pairs_map = generate_pairs_map(unit_geneset_map, geneset_id_map)
    coherence = {}
    for unit in tqdm(unit_geneset_pairs_map):
        coherences = []
        for pair in unit_geneset_pairs_map[unit]:
            if len(pair)>1:
                try:
                    coher = r_go_sim(pair[0], pair[1])
                    print(r_go_sim.cache_info())
                    coherences.append(coher)
                except rpy2.rinterface.RRuntimeError:
                    pass
            else:
                coherences.append(1.0)

        coherence[unit] = np.mean(coherences)


    return coherence
'''

def write_coherence_to_file(path, coherence, unit_geneset_map):
    with open(path+coherence_filename, 'w') as file:
        table = csv.writer(file)
        for unit in coherence:
            if not math.isnan(coherence[unit]):
                table.writerow((unit, coherence[unit], len(unit_geneset_map[unit]), unit_geneset_map[unit]))

def load_coherence_file(path):
    coherence_file = path+coherence_filename
    coherence = {}

    with open(coherence_file, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            unit = int(line[0])
            value = float(line[1])
            coherence[unit]=value

    return coherence

def get_coherence(path, use_web=False):

    go_terms_filename = path+"go_terms.txt"
    term_map_filename = path+"go_term_map.txt"
    biology_filename = path+"biology.txt"
    unit_geneset_map = get_result_file_dict(biology_filename)
    unit_geneset_map = remove_non_GO_terms(unit_geneset_map)

    if os.path.isfile(path+coherence_filename):
        coherence = load_coherence_file(path)
    else:
        term_id_map = build_map(go_terms_filename, term_map_filename, use_web)
        setup_for_r()
        coherence = compute_mean_coherence(unit_geneset_map, term_id_map)
        # Write coherence to file
        write_coherence_to_file(path, coherence, unit_geneset_map)

    return coherence, unit_geneset_map

def remove_non_GO_terms(unit_geneset_map):
    for unit in unit_geneset_map:
        geneset_list = []
        for geneset in unit_geneset_map[unit]:
            if 'GO_' in geneset:
                geneset_list.append(geneset)
        unit_geneset_map[unit] = geneset_list
    return unit_geneset_map

def main(path):

    coherence, unit_geneset_map = get_coherence(path, use_web=False)

    coherence_list = []
    for unit in coherence:
        if not math.isnan(coherence[unit]):
            print(unit_geneset_map[unit])
            print('Coherence {}'.format(coherence[unit]))
            coherence_list.append(coherence[unit])

    print('Coherence mean:{}'.format(np.mean(coherence_list)))
    print('Coherence stdv:{}'.format(np.std(coherence_list)))
    print('Number of Coherence values:{}'.format(len(coherence_list)))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "C:\\Users\\pulin\\Projects\\self-taught\\results\\models\\full_data_scaled_best\\"
        path = "/mnt/c/Users/pulin/Projects/self-taught/results/models/full_data_scaled_best/"

    main(path)

