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
from utils.stats import emp_p_value, fdrcorrection

sort_comp_dict = lambda y, by: sorted(y, key=lambda x: (y[x][by], x))
set_diff = namedtuple('set_diff', 'file0 file1 set_')

def get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units=0.1, disp=False,
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
    if isinstance(gsm, tuple):
        activations = get_delta_activations(gsm, datasets, model)
    else:
        activations = get_activations(gsm, datasets, model)
    sorted_units = sorted(enumerate(activations), key=lambda unit: unit[1], reverse=True)
    top_units = [unit+1 for unit, _ in sorted_units[:n_units_selected]]
    # added one to unit number because biology_results file assume units start with 1

    # added one to unit number because biology_results file assume units start with 1
    geneset_list = build_genesetlist_from_units(top_units, geneset_unit_map, disp, bio_result_file)

    return activations, list(set(geneset_list))


def get_ggec(gsm_list, geneset_unit_map, model, datasets,
             for_top_x_pct_units):
    ggec = Counter()
    if isinstance(gsm_list, tuple):
        input_sum1 = np.mean([np.array(get_input(gsm, datasets)) for gsm in gsm_list[0]], axis=0)
        input_sum2 = np.mean([np.array(get_input(gsm, datasets)) for gsm in gsm_list[1]], axis=0)
        model_input = abs(input_sum1 - input_sum2)
        ggec += Counter(get_activated_genesets(model_input, geneset_unit_map, model, datasets,
                                               for_top_x_pct_units)[0])
    else:
        for gsm in gsm_list:
            ggec += Counter(
                get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units)[0])
    return ggec

def get_unit_activation(gsm_list, geneset_unit_map, model, datasets,
             for_top_x_pct_units):
    gsm_encoding = {}
    for gsm in gsm_list:
        activations, genesets = get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units)
        sorted_units = sorted(enumerate(activations), key=lambda unit: unit[1], reverse=True)
        top_units = [unit+1 for unit, _ in sorted_units[:int(len(activations)*for_top_x_pct_units)]]
        gsm_encoding[gsm]=top_units
    return gsm_encoding

def get_monte_carlo_activation(gsm_count, geneset_unit_map, model, datasets,
                               tests=1000):
    random_test = np.zeros(shape=(model.encoding_size,tests))
    for_top_x_pct_units=model.rho
    _ = datasets[0].next_batch(datasets[0].num_examples)
    for i in tqdm.tqdm(range(tests)):
        unit_activations = []
        if isinstance(gsm_count, tuple):
           _, random_gsm_list1 = datasets[0].next_batch(gsm_count[0])
           _, random_gsm_list2 = datasets[0].next_batch(gsm_count[1])
           random_gsm_list = (random_gsm_list1, random_gsm_list2)
        else:
            _, random_gsm_list = datasets[0].next_batch(gsm_count)

        encoding = get_unit_activation(random_gsm_list, geneset_unit_map, model, datasets, for_top_x_pct_units)
        for gsm in encoding:
            unit_activations.extend(encoding[gsm])
        test_stat = Counter(unit_activations)
        for unit in test_stat:
            random_test[unit-1][i]=test_stat[unit]
    return random_test

def main():
    model_folder = os.path.join('results', 'models', 'L1000_scaled_best_3')

    result_filename = 'biology.txt'
    model_name = 'model.net'
    normed_split_path = os.path.join('data','L1000_data','scaled', 'split_')
    split = 1

    labelled_data_files = ['GSE8671_case.txt',
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
                   set_diff(file0=labelled_data_files[0],
                            file1=labelled_data_files[1],
                            #underscore for differentiating between delta tuples
                            set_='_'),
                   # normal tuple for delta_datasets
                   (labelled_data_files[0], labelled_data_files[1]),
                   labelled_data_files[0],
                   labelled_data_files[1],
                   labelled_data_files[2]
                   ]
    '''
    comparision = [ labelled_data_files[5],
                    labelled_data_files[6]
                    ]
    '''
    for_top_x_pct_units=model.rho

    def flattened_active_units(files):
        flat_units=[]
        for file in files:
            encodings=get_unit_activation(gsm_labels[file],geneset_unit_map,model,dataset,for_top_x_pct_units)
            print(len(encodings))
            for gsm in encodings:
                flat_units.extend(encodings[gsm])
        return flat_units

    def plot_files_encoding(files):
        n_units = 2000
        titles=['GSE8671 Control', 'GSE8671 Case']
        import matplotlib.pyplot as plt
        flat_units = flattened_active_units([files[0]])
        control_freq = np.zeros((n_units,))
        for unit in flat_units:
            control_freq[unit-1] += 1
        plt.figure(1)
        plt.title('GSE8671')
        plt.subplot(211)
        plt.ylabel('Frequency of Activation')
        plt.bar(range(n_units),control_freq, label='Control')

        flat_units = flattened_active_units([files[1]])
        case_freq = np.zeros((n_units,))
        for unit in flat_units:
            case_freq[unit-1] += 1
        plt.subplot(212)
        plt.bar(range(n_units),case_freq, label='Control')
        plt.legend(labels=['Control', 'Case'])
        plt.xlabel('Units')
        plt.show()
        with open(os.path.join(model_folder,'activations_'+files[0]+'_'+files[1]+'.csv'), 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Unit',files1[0], 'mean', 'std', files2[0], 'mean', 'std'])
            test_stats1 = get_monte_carlo_activation(len(gsm_labels[files[0]]), geneset_unit_map, model, dataset)
            test_stats2 = get_monte_carlo_activation(len(gsm_labels[files[1]]), geneset_unit_map, model, dataset)
            for row in zip(range(n_units), control_freq,  np.mean(test_stats1, axis=1), np.std(test_stats1, axis=1),
                           case_freq, np.mean(test_stats2, axis=1), np.std(test_stats2, axis=1)):
                writer.writerow(row)
            unit_stats = {}

        p_values = [0]*n_units
        for row in zip(range(2000), test_stats1, test_stats2):
            mean = np.mean(row[1]-row[2])
            std = np.std(row[1]-row[2])
            value = case_freq[row[0]]-control_freq[row[0]]
            p_value = emp_p_value(value, mean, std)

            unit_stats[row[0]] = {'mean': mean,
                                  'std': std,
                                  'p': p_value}
            p_values[row[0]]=p_value

        corrected_p_values = fdrcorrection(p_values)
        for unit in unit_stats:
            unit_stats[unit]['fdr']=corrected_p_values[1][unit]

        print('Gene sets in enriched units')
        genesets = []
        unit_count = 0
        for unit in unit_stats:
            if unit_stats[unit]['fdr']<0.05:
                unit_count+=1
                genesets.extend(geneset_unit_map[unit])

        for geneset in genesets:
            print(geneset)

        print('Enriched units ', unit_count)

    def enriched_units(files):
        group_encoding = Counter(flattened_active_units(files))
        print(group_encoding)
        print(sorted(group_encoding, key=lambda x: group_encoding[x], reverse=True))
        total_samples = sum([len(gsm_labels[file]) for file in files])
        enriched_units = [unit for unit in group_encoding if group_encoding[unit] > total_samples / 2]
        return enriched_units


    files1 = labelled_data_files[0:1]
    files2 = labelled_data_files[1:2]
    plot_files_encoding([files1[0],files2[0]])


if __name__ == '__main__':

    main()