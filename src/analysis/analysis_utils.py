import os
import re
import pickle as pkl
import numpy as np
from collections import defaultdict
from src.utils import ae, get_gsm_labels
from functools import lru_cache

def print_unit_biology(unit_num, biology_result_file):
    with open(biology_result_file) as f:
        for line in f:
            if 'unit' + str(unit_num) + '_gsa' in line:
                print('unit ' + str(unit_num))
                while True:
                    line = f.readline()
                    print(line, end='')
                    if line == '\n' or line == '':
                        break
                break

def get_delta_activations(gsms, datasets, model):
    input_genes0 = get_input(gsms[0], datasets)
    input_genes1 = get_input(gsms[1], datasets)
    if input_genes0 is not None and input_genes1 is not None:
        delta_input = input_genes0-input_genes1
        model_input = np.expand_dims(delta_input, axis=0)
        activations = model.encoding(model_input)[0]
    return activations

def get_activations(gsm, datasets, model):
    input_genes = get_input(gsm, datasets)
    if input_genes is not None:
        model_input = np.expand_dims(input_genes, axis=0)
        activations = model.encoding(model_input)[0]
    return activations

def build_genesetlist_from_units(units, geneset_unit_map, display=False, bio_result_file=''):
    geneset_list = []
    for unit in units:
        geneset_list.extend(geneset_unit_map[unit])
        if display and bio_result_file != '':
            print_unit_biology(unit, bio_result_file)
    return geneset_list

@lru_cache(maxsize=10)
def get_indexed_label_hashmap(dataset, epochs):
    return dict(zip(dataset.labels.tolist(), range(len(dataset.labels))))

def get_input(gsm, datasets):
    data = None
    for dataset in datasets:
        labels_map = get_indexed_label_hashmap(dataset, dataset.epochs_completed)
        if gsm in labels_map:
            data = dataset.images[labels_map[gsm]]
    if data is None:
       print('GSM not found in any dataset')
    return data

def get_result_file_dict(biology_result_file):
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

def extract_labels(labelled_data_files, datasets):
    labelled_data_folder = os.path.join('data')
    # labelled_data_files = ['GSE8052_asthma_1.txt', 'GSE8052_asthma_0.txt']
    # labelled_data_files = ['GDS4602_3.txt', 'GDS4602_4.txt']
    gsm_labels = get_gsm_labels(labelled_data_files, labelled_data_folder)
    for filename in gsm_labels:
        print(filename)
        print("size before:", len(gsm_labels[filename]))
        for i, gsm in enumerate(gsm_labels[filename]):
            if get_input(gsm, datasets=datasets) is None:
                gsm_labels[filename].pop(i)
        print("size after:", len(gsm_labels[filename]))
    return gsm_labels

def setup_analysis(labelled_data_files, model_folder, normed_split_path, model_name='model.net', result_filename='biology.txt', split=1):
    model_file = os.path.join(model_folder, model_name)
    model = ae.Autoencoder.load_model(model_file, logdir=os.path.join('results', 'features_' + model_name))
    print("using " + result_filename)
    bio_result_file = os.path.join(model_folder, result_filename)
    unlabelled, labelled, validation, test = pkl.load(open(normed_split_path + str(split) + '.pkl', 'rb'))
    dataset = [unlabelled, validation]
    gsm_labels = extract_labels(labelled_data_files, dataset)
    geneset_unit_map = get_result_file_dict(bio_result_file)
    return dataset, geneset_unit_map, gsm_labels, model
