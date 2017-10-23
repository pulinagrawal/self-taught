import os
import argparse
import numpy as np
import pickle as pkl

from utils import ae
from utils import ffd
import utils
import trainer
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
#from src.utils import image as im

def build_dataset(features, labels):
    dataset = DataSet(features, labels, reshape=False)
    dataset._images = features
    return dataset

def get_gsm_list(filename):
    gsmlist = []
    with open(filename, 'rb') as gsmlistfile:
        for row in gsmlistfile:
            gsmlist.append(row)

def pull_from_subframe(dataframe, sub_frame):
    data = sub_frame.as_matrix()
    labels = np.array(list(sub_frame.index))
    dataframe.drop(labels, inplace=True)
    return data, labels

def split_dataframes(geodb_dataframe, validation_pct=.05, test_gsm_list_or_pct=None):
    n_samples = geodb_dataframe.shape[0]
    n_validation_samples = int(n_samples*validation_pct)

    if test_gsm_list_or_pct is not None:
        if isinstance(test_gsm_list_or_pct, list):
            test_frame = geodb_dataframe.loc[test_gsm_list_or_pct]
        elif isinstance(test_gsm_list_or_pct, float) and 0 < test_gsm_list_or_pct < 1:
            test_frame = geodb_dataframe.sample(int(n_samples*test_gsm_list_or_pct))

        test = build_dataset(*pull_from_subframe(geodb_dataframe, test_frame))

    validation_frame = geodb_dataframe.sample(n_validation_samples)
    print(validation_frame.head(100))
    #im.plot_genes(validation_frame.head(100).as_matrix())
    validation = build_dataset(*pull_from_subframe(geodb_dataframe, validation_frame))

    training = build_dataset(*pull_from_subframe(geodb_dataframe, geodb_dataframe))

    if test_gsm_list_or_pct is not None:
        return training, validation, test
    else:
        return training, validation


def create_datasets(geodb_filepath, from_pickle=True):

    if from_pickle:
        geo_df = pkl.load(open(geodb_filepath, 'rb'))

    geo_df = geo_df.dropna(axis=0)
    unlabelled, validation = split_dataframes(geo_df)
    labelled = DataSet(np.array([0]), np.array([0]), reshape=False)
    test = DataSet(np.array([0]), np.array([0]), reshape=False)

    return unlabelled, labelled, validation, test

def run_trainer():
    geodb_trainer = trainer.SelfTaughtTrainer()

if __name__ == '__main__':

    split = True
    normed_split_path = os.path.join('data', 'normd_split_')
    parser = argparse.ArgumentParser(description='run on hyperparameters')
    parser.add_argument('--nHidden', type=int, help='number of units in hidden layer')
    parser.add_argument('--learning_rate', type=float, help='learning rate for unsupervised step')
    parser.add_argument('--sparsity', type=float, help='sparsity of the hidden layer')
    parser.add_argument('--beta', type=float, help='value of beta to control the weight of sparsity constraint in loss')
    parser.add_argument('--split', type=int, help='the data split index to use')
    args = parser.parse_args()
    
    if not split:
        normed_path = os.path.join('data', 'transp_normd_1norm.pkl')
        split_data = create_datasets(normed_path)
        pkl.dump(split_data, open(normed_split_path, 'wb'))
        exit()

    #unlabelled, labelled, validation, test = create_datasets(os.path.join(os.pardir, 'data', 'final_transp_directpkl.pkl'))
    #datasets = create_datasets(os.path.join(os.pardir, 'data', 'tranposed_direct.pkl'))
    #pkl.dump(datasets, open('..\\data\\datasets_dropped.pkl', 'wb'))
    # Loading from splits is needed because of different version of pandas on hpc

    unlabelled, labelled, validation, test = pkl.load(open(normed_split_path+str(args.split)+'.pkl', 'rb'))
    input_size = unlabelled._images.shape[1]

    print(args)

    logdir = utils.results_timestamp_dir()
    geo_trainer = trainer.SelfTaughtTrainer(ae.Autoencoder([input_size, args.nHidden],
                                                           learning_rate=args.learning_rate,
                                                           sparsity=args.sparsity,
                                                           beta=args.beta,
                                                           logdir=logdir,
                                                           keep_prob=1

                                                           ),
                                            ffd.FeedForwardNetwork([args.nHidden, 10],),
                                            batch_size=128,
                                            unlabelled=unlabelled,
                                            labelled=labelled,
                                            validation=validation,
                                            test=test,
                                            save_filename='geodb',
                                            run_folder=logdir
                                            )

    geo_trainer.run_unsupervised_training()
    print(logdir)