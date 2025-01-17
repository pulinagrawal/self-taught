import numpy as np
import os
import csv
import pickle as pkl
import pandas as pd
from src.utils import ae

def get_top_features(features, std_away=2, top_x_pct=None):
    if top_x_pct is not None:
        k = int(len(features)*top_x_pct)
        ids = np.argpartion(features, k)[-k:]
        return ids

    std_away = np.std(features)*std_away
    ids = [i for i in range(len(features)) if features[i] > std_away or features[i] < -std_away]

    return ids

def get_gene_list(data_file):
    dataset = pkl.load(open(data_file, 'rb'))
    # geodb_training used 'transp_normd_1norm.pkl', does not matter, no genes removed between the steps
    return dataset.columns

def get_features(model_folder, data_file, model_name='model.net'):
    model_file = os.path.join(model_folder, model_name)
    model = ae.Autoencoder.load_model(model_file, logdir=os.path.join('/mnt/c/Users/pulin/Projects/self-taught/results', 'features_'+model_name))

    features = pd.DataFrame.from_records(model.weights[0])
    genelist = get_gene_list(data_file)
    features = features.set_index(genelist)

    for idx in features.iterrows():
        geneids = str(idx[0]).split(' /// ')
        if len(geneids) > 1:
            features = features.drop(idx[0])

    return features

if __name__ == '__main__':

    model_name = 'geodb_ae_89.net'
    data_file = os.path.join('/mnt/c/Users/pulin/Projects/self-taught/data', 'transp_normd_1norm.pkl')
    model_folder = os.path.join('/mnt/c/Users/pulin/Projects/self-taught/results', 'best_attmpt_2')
    output_file = os.path.join(model_folder, 'features_'+model_name.split('.')[0]+'.csv')

    features = get_features(model_folder, data_file, model_name)

    print(features.shape)
    features.to_csv(output_file, sep='\t')

