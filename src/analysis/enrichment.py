from src.utils.stats import hypgeom_mean, hypgeom_var, hypgeom_pmf
from src.utils import stats, reverse_dict_of_lists
from collections import defaultdict
from collections import Counter
from functools import partial
from bisect import bisect
from src.analysis import setup_analysis, build_genesetlist_from_units, get_activations
import numpy as np
import os
import tqdm

def get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units=0.1, random=False, disp=False,
                 bio_result_file=''):
    """Returns a list of genesets with their respective counts of the times
        they showed up in the model with the given biology_hash (a dictionary of units with
        their corresponding enriched genesets) for thetop x% units.
    :param gsm:  a gsm in datasets
    :param geneset_unit_map:
    :param model:
    :param datasets:
    :param for_top_x_pct_units:
    :param random:
    :param disp:
    :param bio_result_file:
    :return:
    """
    n_units_selected = int(model.encoding_size * for_top_x_pct_units)
    if not random:
        activations = get_activations(gsm, datasets, model)
        sorted_units = sorted(enumerate(activations), key=lambda unit: unit[1], reverse=True)
        top_units = [unit for unit, _ in sorted_units[:n_units_selected]]
        # added one to unit number because biology_results file assume units start with 1
    else:
        top_units = np.random.randint(0, model.encoding_size, size=n_units_selected)
        # added one to unit number because biology_results file assume units start with 1

    top_units = [unit + 1 for unit in top_units]
    geneset_list = build_genesetlist_from_units(top_units, geneset_unit_map, disp, bio_result_file)

    return list(set(geneset_list))


def get_ggec(gsm_list, geneset_unit_map, model, datasets, for_top_x_pct_units):
    ggec = Counter()
    for gsm in gsm_list:
        ggec += Counter(get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units))
    return ggec


def get_monte_carlo_pvalues(gsm_count, geneset_unit_map, model, datasets, for_top_x_pct_units, hypgeoK_geneset_map, tests=30):
    empty_list = partial(list, [0] * tests)
    random_test_dict = defaultdict(empty_list)
    _ = datasets[0].next_batch(datasets[0].num_examples)
    for i in tqdm.tqdm(range(tests)):
       _, random_gsm_list = datasets[0].next_batch(gsm_count)
       ggec = get_ggec(random_gsm_list, geneset_unit_map, model, datasets, for_top_x_pct_units)
       for geneset in ggec:
           random_test_dict[geneset][i] = ggec[geneset]

    montcarl_pvalue = []
    unit_geneset_map = reverse_dict_of_lists(geneset_unit_map)

    n = for_top_x_pct_units * model.encoding_size
    N = model.encoding_size
    random_comp_dict = defaultdict(dict)

    for geneset in tqdm.tqdm(unit_geneset_map):
        '''
        plot = plt.figure(1)
        plt.hist(random_test_dict[geneset])
        plt.show()
        '''
        K = hypgeoK_geneset_map[geneset]
        theoretical_mean = gsm_count * hypgeom_mean(n, K, N)  # len(gsm_list) for sum of hypgeom
        theoretical_var = gsm_count * hypgeom_var(n, K, N)
        random_comp_dict[geneset]["emp_mean"] = np.mean(random_test_dict[geneset])
        random_comp_dict[geneset]["emp_std"] = np.std(random_test_dict[geneset])

        random_comp_dict[geneset]["th_mean"] = theoretical_mean
        random_comp_dict[geneset]["th_std"] = theoretical_var ** 0.5

        # get p-value for each geneset for each random test
        if geneset in random_test_dict:
            for value in random_test_dict[geneset]:
                montcarl_pvalue.append(stats.emp_p_value(value, theoretical_mean, theoretical_var ** 0.5))
        else:
            # 1 is added to pvalue for better approximation suggested in (North et al, 2002)
            for _ in range(tests):
                montcarl_pvalue.append(1.0)

    for geneset in random_comp_dict:
        print(geneset)
        print("th:{0}+{1}".format(random_comp_dict[geneset]["th_mean"], random_comp_dict[geneset]["th_std"]))
        print("emp:{0}+{1}".format(random_comp_dict[geneset]["emp_mean"], random_comp_dict[geneset]["emp_std"]))

    montcarl_pvalue.sort()

    return montcarl_pvalue, random_comp_dict


def enrichment_ref(gsm_list, model, geneset_unit_map, for_top_x_pct_units, datasets, display_units_for_gsms=False,
                   bio_result_file=''):
    unit_geneset_map = reverse_dict_of_lists(geneset_unit_map)
    hypgeoK_geneset_map = {geneset: len(unit_geneset_map[geneset]) for geneset in unit_geneset_map}
    n = for_top_x_pct_units * model.encoding_size
    N = model.encoding_size

    print('Running Monte Carlo Simulation with binomial theoretical')
    montecarlo_pvalues, random_dict = get_monte_carlo_pvalues(len(gsm_list), geneset_unit_map, model, datasets, for_top_x_pct_units,
                                                hypgeoK_geneset_map, tests=100)

    ggec = get_ggec(gsm_list, geneset_unit_map, model, datasets, for_top_x_pct_units)

    comparision_dict = defaultdict(dict)

    # get the GGEC for each geneset from the counter
    for geneset, freq in ggec.items():
        comparision_dict[geneset]['ggec'] = freq
        K = hypgeoK_geneset_map[geneset]
        binom_p = hypgeom_pmf(1, n, K, N)
        theoretical_mean = len(gsm_list) * binom_p # len(gsm_list) for sum of hypgeom
        theoretical_var = len(gsm_list) * (1-binom_p) * binom_p

        pvalue = stats.emp_p_value(comparision_dict[geneset]['ggec'], random_dict[geneset]["emp_mean"], random_dict[geneset]["emp_std"])
        comparision_dict[geneset]['th_avg_gec'] = theoretical_mean
        comparision_dict[geneset]['th_std_gec'] = theoretical_var ** 0.5
        comparision_dict[geneset]['emp_avg_gec'] = random_dict[geneset]['emp_mean']
        comparision_dict[geneset]['emp_std_gec'] = random_dict[geneset]['emp_std']
        comparision_dict[geneset]['obs_pvalue'] = pvalue

    print("Computing FDR scores")
    for geneset in tqdm.tqdm(comparision_dict):
        count = bisect(montecarlo_pvalues, comparision_dict[geneset]['obs_pvalue'])+1
        # probability of getting obs_pvalue or smaller
        comparision_dict[geneset]['fdr'] = count/len(montecarlo_pvalues)

    return comparision_dict

def enrichment(gsm_list, model, geneset_unit_map, for_top_x_pct_units, datasets, display_units_for_gsms=False,
               bio_result_file=''):

    # from the map of genesets relevant to a unit get K for each geneset's hypergeometric distribution
    unit_geneset_map = reverse_dict_of_lists(geneset_unit_map)
    hypgeoK_geneset_map = {geneset: len(unit_geneset_map[geneset]) for geneset in unit_geneset_map}

    # Run Monte Carlo Simulations
    random_tests_count = 10
    empty_list = partial(list, [0] * random_tests_count)
    random_test_dict = defaultdict(empty_list)
    print('Running Monte Carlo Simulation')
    # 1 - shuffle gsm list
    _ = datasets[0].next_batch(datasets[0].num_examples)

    # 2 - run 'random_tests_count' number of random tests
    for i in tqdm.tqdm(range(random_tests_count)):
        test_biology = Counter()
        # 2a - pick pre-determined number of random gsms
        _, random_gsm_list = datasets[0].next_batch(len(gsm_list))
        for gsm in random_gsm_list:
            test_biology += Counter(get_activated_genesets(gsm, geneset_unit_map, model,
                                                 datasets,
                                                 for_top_x_pct_units=for_top_x_pct_units)
                                    )
        for geneset in test_biology:
            random_test_dict[geneset][i] = [test_biology[geneset]]

    max_tests = max([len(random_test_dict[geneset]) for geneset in random_test_dict])
    print('Running Enrichment with {0} Monte Carlo Simulations'.format(max_tests))
    #TODO reduce gsm_list for iunavailable gsms
    # we have a set of GSMs
    gsm_list_biology = []
    for gsm in gsm_list:
        # collect all the genesets enriched in a GSM for all GSMs
        gsm_list_biology.extend(get_activated_genesets(gsm, geneset_unit_map, model, datasets,
                                             for_top_x_pct_units=for_top_x_pct_units,
                                             disp=display_units_for_gsms,
                                             bio_result_file=bio_result_file))

    # structure { geneset_name: {'gec', 'th_avg_gec', 'th_std_gec', 'obs_pvalue', 'fdr'} }
    comparision_dict = defaultdict(dict)

    # get the GGEC for each geneset from the counter
    for geneset, freq in sorted(Counter(gsm_list_biology).items(), key=lambda x: x[1], reverse=True):
        comparision_dict[geneset]['ggec'] = freq

    montcarl_pvalue = []
    print()
    print('Computing p-values')
    for geneset in tqdm.tqdm(unit_geneset_map):
        '''
        plot = plt.figure(1)
        plt.hist(random_test_dict[geneset])
        plt.show()
        '''
        K = hypgeoK_geneset_map[geneset]
        n = for_top_x_pct_units * model.encoding_size
        N = model.encoding_size
        theoretical_mean = len(gsm_list) * hypgeom_mean(n, K, N)  # len(gsm_list) for sum of hypgeom
        theoretical_var = len(gsm_list) * hypgeom_var(n, K, N)

        # get p-value for each geneset for each random test
        if geneset in random_test_dict:
            for value in random_test_dict[geneset]:
                montcarl_pvalue.append(stats.emp_p_value(value, theoretical_mean, theoretical_var ** 0.5))
        else:
            # 1 is added to pvalue for better approximation suggested in (North et al, 2002)
            for _ in range(random_tests_count):
                montcarl_pvalue.append(1.0)

        # get p-value for each geneset for each observation
        if geneset in comparision_dict:
            pvalue = stats.emp_p_value(comparision_dict[geneset]['ggec'], theoretical_mean, theoretical_var ** 0.5)
            comparision_dict[geneset]['th_avg_gec'] = theoretical_mean
            comparision_dict[geneset]['th_std_gec'] = theoretical_var ** 0.5
            comparision_dict[geneset]['obs_pvalue'] = pvalue

    montcarl_pvalue.sort()
    # FDR correction
    print("Computing FDR scores")
    for geneset in tqdm.tqdm(comparision_dict):
        count = bisect(montcarl_pvalue, comparision_dict[geneset]['obs_pvalue'])+1
        # probability of getting obs_pvalue or smaller
        comparision_dict[geneset]['fdr'] = count/len(montcarl_pvalue)

    return comparision_dict



def count_greater(arr, k):
    i = 0
    for v in arr:
        if v > k:
            i += 1
    return i


def main():
    model_name = 'geodb_ae_89.net'
    model_folder = os.path.join('results', 'best_attmpt_2')

    labelled_data_files = ['GSE15061_aml.txt', 'GSE15061_mds.txt']

    result_filename = 'best_sparse_biology.txt'

    normed_split_path = os.path.join('data', 'normd_split_')
    split = 1

    for_top_x_pct_units = 0.02

    dataset, geneset_unit_map, gsm_labels, model = setup_analysis(labelled_data_files, model_folder, model_name,
                                                               normed_split_path, result_filename, split)

    '''
    _ = unlabelled.next_batch(unlabelled.num_examples)
    _, gsm_labels[0] = unlabelled.next_batch(len(gsm_labels[0]))
    _, gsm_labels[1] = unlabelled.next_batch(len(gsm_labels[1]))
    labelled_data_files = ['Random1', 'Random2']
    '''
    comparision_dict_0 = enrichment_ref(gsm_labels[0], model, geneset_unit_map, for_top_x_pct_units, dataset)
    print(labelled_data_files[0]+" genesets")
    print_comp_dict(comparision_dict_0)
    set1 = get_really_enriched(comparision_dict_0)

    print("")

    comparision_dict_1 = enrichment_ref(gsm_labels[1], model, geneset_unit_map, for_top_x_pct_units, dataset)
    print(labelled_data_files[1]+" genesets")
    print_comp_dict(comparision_dict_1)
    set2 = get_really_enriched(comparision_dict_1)

    common = set1.intersection(set2)
    set1_unique = set1-set2
    set2_unique = set2-set1

    print_list = lambda x: ('{}\n'*len(x)).format(*x)
    print('Common')
    print(print_list(common))

    print(labelled_data_files[0]+' Unique')
    print(print_list(sorted(set1_unique)))
    print(labelled_data_files[1]+' Unique')
    print(print_list(sorted(set2_unique)))

    result_path = os.path.join('results')
    filename_dict = defaultdict(list)
    filename_dict[1] = [common, labelled_data_files[0]+'_vs_'+labelled_data_files[1]+'.csv', comparision_dict_0]
    filename_dict[2] = [set1_unique, labelled_data_files[0]+'.csv', comparision_dict_0]
    filename_dict[3] = [set2_unique, labelled_data_files[1]+'.csv', comparision_dict_1]
    import csv

    for filename_set in filename_dict:
        with open(os.path.join(result_path, filename_dict[filename_set][1]), 'w') as csvfile:
            writer = csv.writer(csvfile)
            for geneset in filename_dict[filename_set][0]:
                gene = geneset
                fdr = filename_dict[filename_set][2][geneset]['fdr']
                writer.writerow([geneset, fdr])


sort_comp_dict = lambda y, by: sorted(y, key=lambda x: (y[x][by], x))


def get_really_enriched(comparision_dict):
    return set([geneset for geneset in sort_comp_dict(comparision_dict, 'fdr') if comparision_dict[geneset]['fdr'] < 0.05])


def print_comp_dict(comparision_dict):
    print()
    for item in sort_comp_dict(comparision_dict, 'fdr'):
        value = comparision_dict[item]['ggec']
        mean = comparision_dict[item]['emp_avg_gec']
        std = comparision_dict[item]['emp_std_gec']
        pvalue = comparision_dict[item]['obs_pvalue']
        fdr = comparision_dict[item]['fdr']
        print(item, ':', value, '\t', round(mean, 3), '±', round(std, 3), 'p:', pvalue, 'fdr:', fdr)

if __name__ == '__main__':

    main()