from src.utils.stats import hypgeom_mean, hypgeom_var, hypgeom_pmf, fdrcorrection
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

sort_comp_dict = lambda y, by: sorted(y, key=lambda x: (y[x][by], x))
set_diff = namedtuple('set_diff', 'file0 file1 set_')

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
                            hypgeoK_geneset_map, tests):
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
        binom_p = hypgeom_pmf(1, n, K, N)
        theoretical_mean = gsm_count * binom_p # len(gsm_list) for sum of hypgeom
        theoretical_var = gsm_count * (1-binom_p) * binom_p

        random_comp_dict[geneset]["emp_mean"] = np.mean(random_test_dict[geneset]) \
                                                if geneset in random_test_dict \
                                                else 0
        random_comp_dict[geneset]["emp_std"] = np.std(random_test_dict[geneset]) \
                                               if geneset in random_test_dict \
                                               else 0

        random_comp_dict[geneset]["th_mean"] = theoretical_mean
        random_comp_dict[geneset]["th_std"] = theoretical_var ** 0.5

    for geneset in random_comp_dict:
        print(geneset)
        print("th:{0}+{1}".format(random_comp_dict[geneset]["th_mean"],
                                  random_comp_dict[geneset]["th_std"]))
        print("emp:{0}+{1}".format(random_comp_dict[geneset]["emp_mean"],
                                   random_comp_dict[geneset]["emp_std"]))

    return random_comp_dict

def enrichment_ref(gsm_list, model, geneset_unit_map, for_top_x_pct_units, datasets, display_units_for_gsms=False,
                   bio_result_file=''):
    unit_geneset_map = reverse_dict_of_lists(geneset_unit_map)
    hypgeoK_geneset_map = {geneset: len(unit_geneset_map[geneset]) for geneset in unit_geneset_map}
    n = for_top_x_pct_units * model.encoding_size
    N = model.encoding_size

    print('Running Monte Carlo Simulation with binomial theoretical')
    # if gsm list is tuple
    # means set_diff approach
    if isinstance(gsm_list, tuple):
        gsm_count = (len(gsm_list[0]), len(gsm_list[1]))
    # else delta dataset and regular gsm set
    # both need the number of inputs made into model
    else:
        gsm_count = len(gsm_list)

    random_dict = get_monte_carlo_pvalues(gsm_count, geneset_unit_map, model, datasets, for_top_x_pct_units,
                                                hypgeoK_geneset_map, tests=100)

    ggec = get_ggec(gsm_list, geneset_unit_map, model, datasets, for_top_x_pct_units)

    comparision_dict = defaultdict(dict)

    montecarlo_pvalues = []
    geneset_order = []
    # get the GGEC for each geneset from the counter
    for geneset in unit_geneset_map:
        freq = ggec[geneset] if geneset in ggec else 0
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
        montecarlo_pvalues.append(pvalue)
        geneset_order.append(geneset)
        #get pvalues from all the genesets that show up in the network. Not just the ones enriched in the gsm set.:w


    print("Computing FDR scores")
    corrected_p_values = fdrcorrection(montecarlo_pvalues)
    for geneset in tqdm.tqdm(comparision_dict):
        # probability of getting obs_pvalue or smaller
        comparision_dict[geneset]['fdr'] = corrected_p_values[1][geneset_order.index(geneset)]

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
    model_folder = os.path.join('results', 'models', 'L1000_scaled_best_3')

    result_filename = 'biology.txt'
    model_name = 'model.net'
    normed_split_path = os.path.join('data','L1000_data','scaled', 'split_')
    split = 1

    labelled_data_files = [
                           'GSE8671_case.txt',
                           'GSE8671_control.txt',
                           'GSE8671_series_matrix.txt',
                           'GSE8052_asthma_0.txt',
                           'GSE8052_asthma_1.txt',
                           'GSE15061_aml.txt',
                           'GSE15061_mds.txt',
                           'GDS4602_3.txt',
                           'GDS4602_4.txt',
                           'GDS4602_5.txt',
                           ]

    dataset, geneset_unit_map, gsm_labels, model = setup_analysis(labelled_data_files, model_folder,
                                                                  normed_split_path,
                                                                  model_name, result_filename, split)
    comparision = [
                   # normal tuple for delta_datasets
                   #(labelled_data_files[5], labelled_data_files[6]),
                   labelled_data_files[5],
                   labelled_data_files[6]
                   ]
    '''
    comparision = [ labelled_data_files[5],
                    labelled_data_files[6]
                    ]
    '''
    for_top_x_pct_units=model.rho

    for i, file in enumerate(comparision):
        # if comparison element is a simple tuple
        # assume delta dataset and create list of gsm pairs
        if isinstance(file, tuple) and not isinstance(file, set_diff):
            gsm_labels[file] = list(zip(gsm_labels[file[0]],gsm_labels[file[1]]))
        # if comparision element is a set_diff
        # create tuple of lists of gsms
        elif isinstance(file, set_diff):
            gsm_labels[file] = (gsm_labels[file[0]], gsm_labels[file[1]])

    sets = {}

    for file in comparision:
            enriched = enrichment_ref(gsm_labels[file], model,
                                                geneset_unit_map,
                                                for_top_x_pct_units, dataset)
            print(str(file)+" genesets")
            print_comp_dict(enriched)
            if isinstance(file, set_diff):
                gsm_count = 1
            elif isinstance(file, tuple):
                gsm_count = len(gsm_labels[file[0]])
            else:
                gsm_count = len(gsm_labels[file])
            fdr_filter = lambda enriched: dict([(geneset,enriched[geneset]['fdr']) for geneset in sort_comp_dict(enriched, 'fdr') if enriched[geneset]['fdr'] < 0.05])
            count_filter = lambda enriched: dict([(geneset,enriched[geneset]['fdr']) for geneset in sort_comp_dict(enriched, 'ggec')
                                                 if enriched[geneset]['ggec'] > gsm_count/2 and enriched[geneset]['fdr'] < 0.05])
            print(str(file)+" genesets")
            sets[file] = count_filter(enriched)
            print('After count filter')
            print(sets[file])
            sets[file] = fdr_filter(enriched)
            print('After fdr filter')
            print(sets[file])
            print("")
    all = []
    for file in sets:
        print()
        print(file)
        for geneset in sets[file]:
            print(geneset)

    for file in sets:
        all.extend(sets[file].keys())
    all = set(all)
    set_data = {}
    fdr_data = {}
    for geneset in all:
        set_data[geneset] = [ 1 if geneset in sets[file].keys() else 0 for file in sets ]
        fdr_data[geneset] = [ sets[file][geneset] if geneset in sets[file].keys() else 'x' for file in sets ]

    with open(os.path.join(model_folder,'comparison'+comparision[-1]+'_'+comparision[-2]+'.csv'), 'w') as datafile:
        writer = csv.writer(datafile, delimiter=',')
        header=["Geneset Name"]
        for file in sets:
            if isinstance(file, tuple) and not isinstance(file, set_diff):
                compare_name = file[0]+'-'+file[1]
            elif isinstance(file, set_diff):
                compare_name = '({0})-({1})'.format(file[0], file[1])
            else:
                compare_name = file
            header.append(compare_name)
            header.append(compare_name+'_FDR')
        writer.writerow(header)
        for geneset in set_data:
            rowdat = []
            for presence, fdr in zip(set_data[geneset],fdr_data[geneset]):
                rowdat.extend([presence, fdr])
            writer.writerow([geneset, *rowdat])

    meta = {}
    meta["file"]="https://raw.githubusercontent.com/pulinagrawal/self-taught/upset/results/best_attmpt_2/comparison.csv"
    meta["name"]="Genset Comparison"
    meta["header"]=0
    meta["separator"]=","
    meta["skip"]=0
    meta["meta"]=[{"type":"id", "index":0, "name": "Geneset Name"}]
    meta["sets"]=[{"format":"binary", "start":1, "end": len(comparision)}]
    meta_json = json.dumps(meta)

    with open(os.path.join(model_folder,'comparison.json'), 'w') as jsonfile:
        jsonfile.write(meta_json)

if __name__ == '__main__':

    main()