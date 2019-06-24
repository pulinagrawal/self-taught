import os
import numpy as np
import pickle as pkl
import pandas as pd
import src.utils as utils
from src import trainer
from src.utils import ae
from src.utils import ffd #4mae as ffd
from collections import namedtuple
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def get_gsm_labels(file_list, folder):
    gsm_labels = {}
    i = 0
    for file in file_list:
        with open(os.path.join(folder, file)) as f:
            header = f.readline()
            header = header.split('\n')[0]
            gsms = header.split('\t')[1:]
            gsm_labels[i] = gsms
            i = i+1
    return gsm_labels

def one_hot(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    num_labels = len(set(labels.data))
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

def extract_gsm_data(dataset :DataSet, gsm_labels :dict):
    labels = [label for label_list in gsm_labels.values() for label in label_list]
    dataset_idxs = [i for label, i in zip(dataset.labels, range(dataset.num_examples)) if label in labels]
    mask = np.zeros(len(dataset.labels), dtype=bool)
    mask[[dataset_idxs]] = True
    data_images = dataset.images[mask,...]
    data_labels = dataset.labels[mask,...]
    mask = np.invert(mask)
    dataset._images = dataset.images[mask,...]
    dataset._labels = dataset.labels[mask,...]
    return data_images, data_labels

def rebuild_datasets(dataset_tuple, gsm_labels :dict, pct_split=(.8,.1,.1)):
    #TODO cleanup by breaking up and running through complete tuple
    data_images, data_labels = extract_gsm_data(dataset_tuple[0], gsm_labels)
    di, dl = extract_gsm_data(dataset_tuple[2], gsm_labels)
    data_images = np.concatenate((data_images, di))
    data_labels = np.concatenate((data_labels, dl))

    rev_gsm_labels = utils.reverse_dict_of_lists(gsm_labels)

    data_labels = np.array([rev_gsm_labels[label][0] for label in data_labels])
    labelled_data = DataSet(data_images, one_hot(data_labels), one_hot=True, reshape=False)
    labelled_data._images = data_images

    _ = labelled_data.next_batch(labelled_data.num_examples)
    shuffled_labelled_data = labelled_data

    n_lablled = int(shuffled_labelled_data.num_examples*pct_split[0])
    n_valid_lab = int(shuffled_labelled_data.num_examples*pct_split[1])
    n_test = shuffled_labelled_data.num_examples-n_lablled-n_valid_lab

    imag, lab = shuffled_labelled_data.next_batch(4)

    unlabelled = dataset_tuple[0]
    validation = dataset_tuple[2]
    labelled = DataSet(*shuffled_labelled_data.next_batch(n_lablled), reshape=False)
    validation_lab = DataSet(*shuffled_labelled_data.next_batch(n_valid_lab), reshape=False)
    test = DataSet(*shuffled_labelled_data.next_batch(n_test), reshape=False)

    return unlabelled, labelled, validation, validation_lab, test

def pull_from_datasets(datasets, gsm_labels :dict):
    data_images, data_labels = extract_gsm_data(datasets[0], gsm_labels)
    for i in range(1, len(datasets)):
        di, dl = extract_gsm_data(dataset_tuple[i], gsm_labels)
        if len(di) !=0:
            data_images = np.concatenate((data_images, di))
            data_labels = np.concatenate((data_labels, dl))

    rev_gsm_labels = utils.reverse_dict_of_lists(gsm_labels)

    data_labels = np.array([rev_gsm_labels[label][0] for label in data_labels])
    labelled_data = DataSet(data_images, data_labels, reshape=False)
    labelled_data._images = data_images

    return labelled_data


def rebuild_datasets_proportionally(dataset_tuple, gsm_labels :dict, pct_split=(.8,.1,.1)):
    #TODO cleanup by breaking up and running through complete tuple

    labelled_datasets = {}
    for i in gsm_labels:
        labelled_datasets[i] = pull_from_datasets(dataset_tuple, {i: gsm_labels[i]})
        _ = labelled_datasets[i].next_batch(labelled_datasets[i].num_examples)    # shuffle

    # For labelled
    data_images, data_labels = labelled_datasets[0].next_batch(int(labelled_datasets[0].num_examples*pct_split[0]))
    for i in range(1, len(gsm_labels)):

        di, dl = labelled_datasets[i].next_batch(int(labelled_datasets[i].num_examples*pct_split[0]))
        data_images = np.concatenate((data_images, di))
        data_labels = np.concatenate((data_labels, dl))

    labelled = DataSet(data_images, one_hot(data_labels), one_hot=True, reshape=False)


    # For validation labelled
    data_images, data_labels = labelled_datasets[0].next_batch(int(labelled_datasets[0].num_examples*pct_split[1]))
    for i in range(1, len(gsm_labels)):

        di, dl = labelled_datasets[i].next_batch(int(labelled_datasets[i].num_examples*pct_split[1]))
        data_images = np.concatenate((data_images, di))
        data_labels = np.concatenate((data_labels, dl))

    validation_lab = DataSet(data_images, one_hot(data_labels), one_hot=True, reshape=False)


    # For test
    n_lablled = int(labelled_datasets[0].num_examples*pct_split[0])
    n_valid_lab = int(labelled_datasets[0].num_examples*pct_split[1])
    n_test = labelled_datasets[0].num_examples-n_lablled-n_valid_lab

    data_images, data_labels = labelled_datasets[0].next_batch(n_test)
    for i in range(1, len(gsm_labels)):

        n_lablled = int(labelled_datasets[i].num_examples*pct_split[0])
        n_valid_lab = int(labelled_datasets[i].num_examples*pct_split[1])
        n_test = labelled_datasets[i].num_examples-n_lablled-n_valid_lab

        di, dl = labelled_datasets[i].next_batch(n_test)
        data_images = np.concatenate((data_images, di))
        data_labels = np.concatenate((data_labels, dl))

    test = DataSet(data_images, one_hot(data_labels), one_hot=True, reshape=False)

    unlabelled = dataset_tuple[0]
    validation = dataset_tuple[2]

    return unlabelled, labelled, validation, validation_lab, test


if __name__ == '__main__':

    labelled_data_folder = os.path.join('data')
    #labelled_data_files = ['GSE8052_asthma_0.txt', 'GSE8052_asthma_1.txt']
    labelled_data_files = ['GSE15061_aml.txt', 'GSE15061_mds.txt']

    model_name = 'geodb_ae_89.net'
    model_folder = os.path.join('results', 'best_attmpt_2')
    model_file = os.path.join(model_folder, model_name)

    logdir = utils.results_timestamp_dir()

    normed_split_path = os.path.join('data', 'normd_split_')
    dataset_tuple = pkl.load(open(normed_split_path+str(1)+'.pkl', 'rb'))

    gsm_labels = get_gsm_labels(labelled_data_files, labelled_data_folder)

    unlabelled, labelled, validation, validation_lab, test = rebuild_datasets_proportionally(dataset_tuple, gsm_labels)

    input_features_size = np.shape(validation_lab.next_batch(validation_lab.num_examples)[0])[1]
    control_model = trainer.SelfTaughtTrainer(ae.Autoencoder.get_identity_encoder(input_features_size),
                                              ffd.FeedForwardNetwork([input_features_size, len(labelled_data_files)]),
                                            batch_size=40,
                                            unlabelled=unlabelled,
                                            labelled=labelled,
                                            validation=validation,
                                            validation_lab=validation_lab,
                                            test=test,
                                            save_filename='geodb_classifier',
                                            run_folder=logdir)

    control_model.build_validation_features()
    print(control_model.run_supervised_training())


    autoenc = ae.Autoencoder.load_model(model_file, logdir=os.path.join('results', 'features_' + model_name))

    case_model = trainer.SelfTaughtTrainer(autoenc, ffd.FeedForwardNetwork([autoenc.encoding_size, len(labelled_data_files)]),
                                            batch_size=40,
                                            unlabelled=unlabelled,
                                            labelled=labelled,
                                            validation=validation,
                                            validation_lab=validation_lab,
                                            test=test,
                                            save_filename='geodb_classifier',
                                            run_folder=logdir)
    case_model.build_validation_features()
    print(case_model.run_supervised_training())

