from src.utils.stats import hypgeom_mean, hypgeom_var, hypgeom_pmf
from src.utils import stats, reverse_dict_of_lists
from collections import defaultdict
from collections import namedtuple
from collections import Counter
from functools import partial
from bisect import bisect
from src.analysis.analysis_utils import setup_analysis, build_genesetlist_from_units, get_activations
from src.analysis.analysis_utils import get_delta_activations, get_activations, get_input
import numpy as np
import os
import tqdm
import csv
import json

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
        if isinstance(gsm, tuple):
            activations = get_delta_activations(gsm, datasets, model)
        else:
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


def get_ggec(gsm_list, geneset_unit_map, model, datasets,
             for_top_x_pct_units):
    ggec = Counter()
    if isinstance(gsm_list, tuple):
        input_sum1 = np.mean([np.array(get_input(gsm, datasets)) for gsm in gsm_list[0]], axis=0)
        input_sum2 = np.mean([np.array(get_input(gsm, datasets)) for gsm in gsm_list[1]], axis=0)
        model_input = abs(input_sum1-input_sum2)
        ggec += Counter(get_activated_genesets(model_input, geneset_unit_map, model, datasets, for_top_x_pct_units))
    else:
        for gsm in gsm_list:
            ggec += Counter(get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units))
    return ggec


def get_monte_carlo_pvalues(gsm_count, geneset_unit_map, model, datasets, for_top_x_pct_units,
                            hypgeoK_geneset_map, tests=30):
    empty_list = partial(list, [0] * tests)
    random_test_dict = defaultdict(empty_list)
    _ = datasets[0].next_batch(datasets[0].num_examples)
    for i in tqdm.tqdm(range(tests)):
        if isinstance(gsm_count, tuple):
           _, random_gsm_list1 = datasets[0].next_batch(gsm_count[0])
           _, random_gsm_list2 = datasets[0].next_batch(gsm_count[1])
           random_gsm_list = (random_gsm_list1, random_gsm_list2)
        else:
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
        if isinstance(gsm_count, tuple):
            gsm_count=1
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

def enrichment_set_diff(gsm_list, model, geneset_unit_map, for_top_x_pct_units, datasets, display_units_for_gsms=False,
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

def enrichment_ref(gsm_list, model, geneset_unit_map, for_top_x_pct_units, datasets, display_units_for_gsms=False,
                   bio_result_file=''):
    unit_geneset_map = reverse_dict_of_lists(geneset_unit_map)
    hypgeoK_geneset_map = {geneset: len(unit_geneset_map[geneset]) for geneset in unit_geneset_map}
    n = for_top_x_pct_units * model.encoding_size
    N = model.encoding_size

    print('Running Monte Carlo Simulation with binomial theoretical')
    if isinstance(gsm_list, tuple):
        gsm_count = (len(gsm_list[0]), len(gsm_list[1]))
    else:
        gsm_count = len(gsm_list)

    montecarlo_pvalues, random_dict = get_monte_carlo_pvalues(gsm_count, geneset_unit_map, model, datasets, for_top_x_pct_units,
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

sort_comp_dict = lambda y, by: sorted(y, key=lambda x: (y[x][by], x))

def print_comp_dict(comparision_dict):
    print()
    for item in sort_comp_dict(comparision_dict, 'fdr'):
        value = comparision_dict[item]['ggec']
        mean = comparision_dict[item]['emp_avg_gec']
        std = comparision_dict[item]['emp_std_gec']
        pvalue = comparision_dict[item]['obs_pvalue']
        fdr = comparision_dict[item]['fdr']
        print(item, ':', value, '\t', round(mean, 3), '±', round(std, 3), 'p:', pvalue, 'fdr:', fdr)

def main():
    model_name = 'geodb_ae_89.net'
    model_folder = os.path.join('results', 'best_attmpt_2')

    labelled_data_files = ['GSE8671_case.txt', 'GSE8671_control.txt', 'GSE8671_series_matrix.txt']
    set_diff = namedtuple('set_diff', 'file0 file1 set_')
    comparision = [set_diff(file0=labelled_data_files[0],
                            file1=labelled_data_files[1],
                            #underscore for differentiating between delta tuples
                            set_='_'),
                   labelled_data_files[0],
                   labelled_data_files[1],
                   labelled_data_files[2],
                   (labelled_data_files[0],labelled_data_files[1])
                   ]

    result_filename = 'best_sparse_biology.txt'
    normed_split_path = os.path.join('data', 'normd_split_')
    split = 1
    for_top_x_pct_units = 0.02


    dataset, geneset_unit_map, gsm_labels, model = setup_analysis(labelled_data_files, model_folder, normed_split_path,
                                                                model_name, result_filename, split)

    for i, file in enumerate(comparision):
        if isinstance(file, tuple) and not isinstance(file, set_diff):
            gsm_labels[file] = list(zip(gsm_labels[file[0]],gsm_labels[file[1]]))
        elif isinstance(file, set_diff):
            gsm_labels[file] = (gsm_labels[file[0]], gsm_labels[file[1]])

    sets = {}

    for file in comparision:
            enriched = enrichment_ref(gsm_labels[file], model,
                                                geneset_unit_map,
                                                for_top_x_pct_units, dataset)
            print(str(file)+" genesets")
            fdr_filter = lambda enriched: set([geneset for geneset in sort_comp_dict(enriched, 'fdr') if enriched[geneset]['fdr'] < 0.05])
            print_comp_dict(enriched)
            sets[file] = fdr_filter(enriched)
            print("")

    all=set.union(*sets.values())
    set_data = {}
    for geneset in all:
        set_data[geneset] = [ 1 if geneset in sets[file] else 0 for file in sets ]

    with open(os.path.join(model_folder,'comparison.csv'), 'w') as datafile:
        writer = csv.writer(datafile, delimiter=';')
        header=["Geneset Name"]
        for file in sets:
            if isinstance(file, tuple) and not isinstance(file, set_diff):
                header.append(file[0]+'-'+file[1])
            elif isinstance(file, set_diff):
                header.append('{{{0}}}-{{{1}}}'.format(file[0], file[1]))
            else:
                header.append(file)
        writer.writerow(header)
        for geneset in set_data:
            writer.writerow([geneset, *set_data[geneset]])

    meta = {}
    meta["file"]="https://raw.githubusercontent.com/pulinagrawal/self-taught/upset/results/best_attmpt_2/comparison.csv"
    meta["name"]="Genset Comparison"
    meta["header"]=0
    meta["separator"]=";"
    meta["skip"]=0
    meta["meta"]=[{"type":"id", "index":0, "name": "Geneset Name"}]
    meta["sets"]=[{"format":"binary", "start":1, "end": len(comparision)+1}]
    meta_json = json.dumps(meta)

    with open(os.path.join(model_folder,'comparison.json'), 'w') as jsonfile:
        jsonfile.write(meta_json)

if __name__ == '__main__':

    main()