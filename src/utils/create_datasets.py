from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import os
import numpy as np
import pickle as pkl


def build_dataset(features, labels):
    dataset = DataSet(features, labels, reshape=False)
    dataset._images = features
    return dataset


def pull_from_subframe(dataframe, sub_frame):
    data = sub_frame.as_matrix()
    labels = np.array(list(sub_frame.index))
    dataframe.drop(labels, inplace=True)
    return data, labels


def split_dataframes(geodb_dataframe, validation_pct=.05, test_gsm_list_or_pct=None):
    n_samples = geodb_dataframe.shape[0]
    n_validation_samples = int(n_samples * validation_pct)

    if test_gsm_list_or_pct is not None:
        if isinstance(test_gsm_list_or_pct, list):
            test_frame = geodb_dataframe.loc[test_gsm_list_or_pct]
        elif isinstance(test_gsm_list_or_pct, float) and 0 < test_gsm_list_or_pct < 1:
            test_frame = geodb_dataframe.sample(int(n_samples * test_gsm_list_or_pct))

        test = build_dataset(*pull_from_subframe(geodb_dataframe, test_frame))

    validation_frame = geodb_dataframe.sample(n_validation_samples)
    print(validation_frame.head(100))
    # im.plot_genes(validation_frame.head(100).as_matrix())
    validation = build_dataset(*pull_from_subframe(geodb_dataframe, validation_frame))

    training = build_dataset(*pull_from_subframe(geodb_dataframe, geodb_dataframe))

    if test_gsm_list_or_pct is not None:
        return training, validation, test
    else:
        return training, validation


def create_datasets(geodb_filepath):
    geo_df = pkl.load(open(geodb_filepath, 'rb'))
    unlabelled, validation = split_dataframes(geo_df)
    labelled = DataSet(np.array([0]), np.array([0]), reshape=False)
    test = DataSet(np.array([0]), np.array([0]), reshape=False)

    return unlabelled, labelled, validation, test


def create_splits(preprocessed_filename, splits=3):
    for split in range(1, splits+1):
        preprocessed_file = os.path.join('data', preprocessed_filename)
        split_data = create_datasets(preprocessed_file)
        split_filename = preprocessed_filename[:-4]+'_split_'+str(split)+'.pkl'
        pkl.dump(split_data, open(split_filename, 'wb'))

