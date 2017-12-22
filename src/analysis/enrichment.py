from src.utils import ae, stats, reverse_list_dict, get_gsm_labels
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from collections import defaultdict
from collections import Counter
import numpy as np
import os
import tqdm
import re
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pickle as pkl


def get_input(gsm, datasets):
    for dataset in datasets:
        for i, label in enumerate(dataset.labels):
            if label == gsm:
                break
        return dataset.images[i]
    return None


def get_result_file_hash(biology_result_file):
    biology = defaultdict(list)
    with open(biology_result_file) as f:
        for line in f:
            if 'unit' in line:
                if '_gsa' in line:
                    unit = int(re.findall(r'\d+', line)[-1])
                    while True:
                        line = f.readline()
                        geneset = line.split('\t')[0]
                        if geneset != '\n' and geneset != '':
                            biology[unit].append(geneset)
                        if line == '\n' or line == '':
                            break
    return biology

def print_unit_biology(unit_num, biology_result_file):
    with open(bio_result_file) as f:
        for line in f:
            if 'unit' + str(unit_num) + '_gsa' in line:
                print('unit ' + str(unit_num))
                while True:
                    line = f.readline()
                    print(line, end='')
                    if line == '\n' or line == '':
                        break
                break


def get_gsm_biology(gsm, biology_hash, model, datasets, for_top_x_pct_units=0.1, random=False, disp=False, bio_result_file=''):
    n_units_selected = int(model.encoding_size * for_top_x_pct_units)
    if not random:
        input_genes = get_input(gsm, datasets)
        if input_genes is not None:
            model_input = np.expand_dims(input_genes, axis=0)
            encoding = model.encoding(model_input)

        sorted_units = sorted(enumerate(encoding[0]), key=lambda unit: unit[1])
        top_units = [unit+1 for unit, _ in sorted_units[-n_units_selected:]]
        # added one to unit number because biology_results file assume units start with 1
    else:
        top_units = np.random.randint(0, model.encoding_size, size=n_units_selected)
        top_units = [unit+1 for unit in top_units]
        # added one to unit number because biology_results file assume units start with 1

    biology = []
    for unit in top_units:
        if disp and bio_result_file != '':
            print_unit_biology(unit, bio_result_file)
        biology.extend(biology_hash[unit])

    return biology

def hypgeom_mean(n, K, N):
    return n*K/N

def hypgeom_var(n, K, N):
    return n*(K/N)*((N-K)/N)*((N-n)/(N-1))

def count_greater(arr, k):
    i = 0
    for v in arr:
        if v > k:
            i += 1
    return i

if __name__ == '__main__':

    model_name = 'geodb_ae_89.net'
    model_folder = os.path.join('results', 'best_attmpt_2')
    model_file = os.path.join(model_folder, model_name)
    result_filename = 'biology_result.txt'
    bio_result_file = os.path.join(model_folder, result_filename)
    for_top_x_pct_units = 0.02
    display_units_for_gsms = False

    labelled_data_folder = os.path.join('data')
    labelled_data_files = ['GSE8052_asthma_1.txt']

    gsm_labels = get_gsm_labels(labelled_data_files, labelled_data_folder)
    gsm_list = gsm_labels[0]

    model = ae.Autoencoder.load_model(model_file, logdir=os.path.join('results', 'features_' + model_name))
    normed_split_path = os.path.join('data', 'normd_split_')
    split = 1
    unlabelled, labelled, validation, test = pkl.load(open(normed_split_path + str(split) + '.pkl', 'rb'))

    biology_hash = get_result_file_hash(bio_result_file)

    gsm_list_biology = []

    for gsm in gsm_list:
        gsm_list_biology.extend(get_gsm_biology(gsm, biology_hash, model, [unlabelled, validation],
                                                for_top_x_pct_units=for_top_x_pct_units,
                                                disp=display_units_for_gsms,
                                                bio_result_file=bio_result_file))

    rev_hash_bio = reverse_list_dict(biology_hash)
    hypgeo_K_bio = {geneset: len(rev_hash_bio[geneset]) for geneset in rev_hash_bio}
    hypgeo_n = for_top_x_pct_units*model.encoding_size

    saved_monte_carlo = True
    random_tests = 10
    random_biology = [[]] * random_tests

    if not saved_monte_carlo:
        print('Running Monte Carlo Simulation')
        # To ensure shuffling
        _ = unlabelled.next_batch(unlabelled.num_examples)
        random_gsm_list = gsm_list

        for i in tqdm.tqdm(range(random_tests)):
            random_biology[i] = Counter()
            _, random_gsm_list = unlabelled.next_batch(len(gsm_list))
            for gsm in random_gsm_list:
                random_biology[i] += Counter(get_gsm_biology(gsm, biology_hash, model,
                                                             [unlabelled, validation],
                                                             for_top_x_pct_units=for_top_x_pct_units)
                                             )


    monte_carlo_filename = 'mc_save_'+model_name+'.pkl'
    random_test_dict = defaultdict(list)
    if not saved_monte_carlo:
        try:
            with open(monte_carlo_filename, 'rb') as f:
                random_test_dict = pkl.load(f)
        except FileNotFoundError:
            print('Existing Monte Carlo Save not found for this model')
            random_test_dict = defaultdict(list)

        for test in random_biology:
            for geneset in test:
                random_test_dict[geneset] += [test[geneset]]

        with open(monte_carlo_filename, 'wb') as f:
            pkl.dump(random_test_dict, f)
    else:
        with open(monte_carlo_filename, 'rb') as f:
            random_test_dict = pkl.load(f)

    max_tests = max([len(random_test_dict[geneset]) for geneset in random_test_dict])
    print('Running Enrichment with {0} Monte Carlo Simulations'.format(max_tests))

    comparision_dict = defaultdict(dict)
    # structure { geneset_name: {'gec', 'th_avg_gec', 'th_std_gec', 'obs_pvalue', 'fdr'} }

    for geneset, freq in sorted(Counter(gsm_list_biology).items(), key=lambda x: x[1], reverse=True):
        comparision_dict[geneset]['gec'] = freq

    montcarl_pvalue = []
    print()
    print('Computing p-values')
    for geneset in tqdm.tqdm(rev_hash_bio):
        '''
        plot = plt.figure(1)
        plt.hist(random_test_dict[geneset])
        plt.show()
        '''
        K = hypgeo_K_bio[geneset]
        n = for_top_x_pct_units*model.encoding_size
        N = model.encoding_size
        theoretical_mean = len(gsm_list)*hypgeom_mean(n, K, N) # len(gsm_list) for sum of hypgeom
        theoretical_var = len(gsm_list)*hypgeom_var(n, K, N)

        if geneset in random_test_dict:
            for value in random_test_dict[geneset]:
                montcarl_pvalue.append(stats.emp_p_value(value, theoretical_mean, theoretical_var**0.5))
        else:
            for _ in range(random_tests):
                montcarl_pvalue.append(1.0)

        # 1 is added to pvalue for better approximation suggested in (North et al, 2002)
        if geneset in comparision_dict:
            pvalue = stats.emp_p_value(comparision_dict[geneset]['gec'], theoretical_mean, theoretical_var**0.5)
            comparision_dict[geneset]['th_avg_gec'] = theoretical_mean
            comparision_dict[geneset]['th_std_gec'] = theoretical_var**0.5
            comparision_dict[geneset]['obs_pvalue'] = pvalue

    for geneset in comparision_dict:
        count = count_greater(montcarl_pvalue, comparision_dict[geneset]['obs_pvalue'])
        # probability of getting obs_pvalue or smaller
        comparision_dict[geneset]['fdr'] = (len(montcarl_pvalue)-count)/len(montcarl_pvalue)

    print()
    print('Really Enriched')
    count = 0
    for item in sorted(comparision_dict, key=lambda x: comparision_dict[x]['obs_pvalue']):
        value = comparision_dict[item]['gec']
        mean = comparision_dict[item]['th_avg_gec']
        std = comparision_dict[item]['th_std_gec']
        pvalue = comparision_dict[item]['obs_pvalue']
        fdr = comparision_dict[item]['fdr']
        print(item, ':', value, '\t', round(mean, 3), '±', round(std, 3), 'p:', pvalue, 'fdr:', fdr)

