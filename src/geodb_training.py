import pandas
from src.utils import ae
from src import trainer
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def build_dataset(features, labels):
    dataset = DataSet(features, labels, reshape=False)
    dataset.images = features
    return dataset

def get_gsm_list(filename):
    gsmlist = []
    with open(filename, 'rb') as gsmlistfile:
        for row in gsmlistfile:
            gsmlist.append(row)

def pull_from_subframe(dataframe, sub_frame):
    data = sub_frame.to_matrix()
    labels = list(sub_frame.index)
    dataframe.drop(labels, inplace=True)
    return data, labels

def split_dataframes(geodb_dataframe, validation_pct=.05, test_gsm_list_or_pct=None):
    n_samples = geodb_dataframe.shape[0]

    if test_gsm_list_or_pct is not None:
        if isinstance(test_gsm_list_or_pct, list):
            test_frame = geodb_dataframe.loc[test_gsm_list_or_pct]
        elif isinstance(test_gsm_list_or_pct, float) and 0<test_gsm_list_or_pct<1:
            test_frame = geodb_dataframe.sample(int(n_samples*test_gsm_list_or_pct))

        test = build_dataset(*pull_from_subframe(geodb_dataframe, test_frame))

    validation_frame = geodb_dataframe.sample(int(n_samples*validation_pct))
    validation = build_dataset(*pull_from_subframe(geodb_dataframe, validation_frame))

    training = build_dataset(*pull_from_subframe(geodb_dataframe, geodb_dataframe))

    if test_gsm_list_or_pct is not None:
        return training, validation, test
    else:
        return training, validation


def create_dataset(geodb_filepath):


def run_trainer():
    geodb_trainer = trainer.SelfTaughtTrainer()