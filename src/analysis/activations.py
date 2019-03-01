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
    geneset_list = build_genesetlist_from_units(top_units, geneset_unit_map, disp, bio_result_file)

    return activations, list(set(geneset_list))


def get_unit_activation(gsm_list, geneset_unit_map, model, datasets,
             for_top_x_pct_units):
    gsm_encoding = {}
    for gsm in gsm_list:
        activations, genesets = get_activated_genesets(gsm, geneset_unit_map, model, datasets, for_top_x_pct_units)
        sorted_units = sorted(enumerate(activations), key=lambda unit: unit[1], reverse=True)
        top_units = [unit+1 for unit, _ in sorted_units[:int(len(activations)*for_top_x_pct_units)]]
        gsm_encoding[gsm]=top_units
    return gsm_encoding



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
        titles=['GSE8671 Control', 'GSE8671 Case']
        import matplotlib.pyplot as plt
        flat_units = flattened_active_units([files[0]])
        control_freq = np.zeros((2000,))
        for unit in flat_units:
            control_freq[unit-1] += 1
        plt.figure(1)
        plt.title('GSE8671')
        plt.subplot(211)
        plt.ylabel('Frequency of Activation')
        plt.bar(range(2000),control_freq, label='Control')

        flat_units = flattened_active_units([files[1]])
        case_freq = np.zeros((2000,))
        for unit in flat_units:
            case_freq[unit-1] += 1
        plt.subplot(212)
        plt.bar(range(2000),case_freq, label='Control')
        plt.legend(labels=['Control', 'Case'])
        plt.xlabel('Units')
        plt.show()
        with open('activations_'+files[0]+'_'+files[1]+'.csv', 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Unit',files1[0],files2[0]])
            for row in zip(range(2000),control_freq, case_freq):
                writer.writerow(row)
        print(files)

    def enriched_units(files):
        group_encoding = Counter(flattened_active_units(files))
        print(group_encoding)
        print(sorted(group_encoding, key=lambda x: group_encoding[x], reverse=True))
        total_samples = sum([len(gsm_labels[file]) for file in files])
        enriched_units = [unit for unit in group_encoding if group_encoding[unit] > total_samples / 2]
        return enriched_units


    files1 = labelled_data_files[6:7]
    files2 = labelled_data_files[5:6]
    plot_files_encoding([files1[0],files2[0]])
    enriched_units1 = enriched_units(files1)
    enriched_units2 = enriched_units(files2)
    remaining_units = set(enriched_units1)-set(enriched_units2)
    flat_genesets = []
    print(files1,'-',files2)
    for unit in remaining_units:
        flat_genesets.extend(geneset_unit_map[unit])

    set1=set(flat_genesets)
    for geneset in set1:
        print(geneset)

    print('Geneset Diff')
    flat_genesets1=[]
    for unit in enriched_units1:
        flat_genesets1.extend(geneset_unit_map[unit])
    flat_genesets2=[]
    for unit in enriched_units2:
        flat_genesets2.extend(geneset_unit_map[unit])

    set2=set(flat_genesets1)-set(flat_genesets2)
    for geneset in set2:
        print(geneset)

    print('Common')
    for geneset in set.intersection(set2,set2):
        print(geneset)

if __name__ == '__main__':

    main()