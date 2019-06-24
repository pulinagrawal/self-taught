from collections import namedtuple
from collections import Counter
from src.analysis.analysis_utils import setup_analysis, build_genesetlist_from_units, get_activations
from src.analysis.analysis_utils import get_delta_activations, get_activations, get_input
from utils.stats import emp_p_value, fdrcorrection
from functools import partial
from scipy.stats import wilcoxon
import numpy as np
import os
import tqdm
import csv

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
                               tests=100):
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
    training_pct = .95
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

    files1 = labelled_data_files[5]
    files2 = labelled_data_files[6]


    X = np.zeros((len(gsm_labels[files1])+len(gsm_labels[files2]), model.encoding_size))
    Y = np.ndarray((len(gsm_labels[files1])+len(gsm_labels[files2]),))


    get_unit_activations = partial(lambda v,w,x,y,z: get_unit_activation(z,v,w,x,y), geneset_unit_map, model,dataset,for_top_x_pct_units)
    sample_no = 0
    for i, encodings in enumerate([get_unit_activations(gsm_labels[files1]), get_unit_activations(gsm_labels[files2])]):
        for gsm in encodings:
            for unit in encodings[gsm]:
                X[sample_no][unit-1] = 1

            Y[sample_no] = i
            sample_no += 1

    from sklearn import svm
    from sklearn.feature_selection import SelectKBest, chi2

    def test_classification(X, Y):
        error = []
        print("")
        indices = list(range(len(Y)))
        np.random.seed(100)
        for _ in range(100):
            np.random.shuffle(indices)
            print(indices)
            training, test = np.split(indices, [int(len(indices)*training_pct)])
            clf = svm.SVC(gamma='scale')
            #X_new = SelectKBest(chi2, k=2).fit_transform(X[np.ix_(training)], Y[np.ix_(training)])
            clf.fit(X[np.ix_(training)], Y[np.ix_(training)])
            error.append(sum(abs(clf.predict(X[np.ix_(test)]) - Y[np.ix_(test)]))/len(test))
        return error


    X_raw = []
    for i, labels in enumerate([gsm_labels[files1], gsm_labels[files2]]):
        for label in labels:
                X_raw.append(get_input(label, dataset))

    x = np.array(X_raw)

    x_errors = test_classification(x,Y)
    X_errors = test_classification(X,Y)

    print(np.mean(x_errors), np.std(x_errors))
    print(np.mean(X_errors), np.std(X_errors))

    print(wilcoxon(x_errors, X_errors, zero_method="wilcox"))

    from matplotlib import pyplot as plt
    fig1, ax1 = plt.subplots()

    ax1.boxplot([x_errors, X_errors])

    plt.show()

if __name__ == '__main__':

    main()