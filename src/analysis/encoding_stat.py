from src.utils import ae
from src.analysis import enrichment as enr
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

def get_top_features(features, std_away=2, top_x_pct=None):
    if top_x_pct is not None:
        k = int(len(features)*top_x_pct)
        ids = np.argpartion(features, k)[-k:]
        return ids

    std_away = np.std(features)*std_away
    ids = [i for i in range(len(features)) if features[i] > std_away or features[i] < -std_away]

    return ids

def get_gene_list():
    dataset = pkl.load(open(os.path.join('data', 'transp_normd.pkl'), 'rb'))
    return dataset.columns

if __name__ == '__main__':

   model_name = 'geodb_ae_91.net'
   model_file = os.path.join('results', 'best_dense_attmpt_1', model_name)
   model = ae.Autoencoder.load_model(model_file, logdir=os.path.join('results', 'features_'+model_name))
   normed_split_path = os.path.join('data', 'normd_split_')
   split = 1
   unlabelled, labelled, validation, test = pkl.load(open(normed_split_path+str(split)+'.pkl', 'rb'))
   print(unlabelled.num_examples)

   encodings = model.encoding(unlabelled.next_batch(unlabelled.num_examples)[0])
   fig = plt.figure(1)
   plt.hist(encodings[0])
   plt.show()

   activity_frequency = np.count_nonzero(encodings>0.18, axis=0)
   print(activity_frequency.shape)

   most_active = [i+1 for i, freq in enumerate(activity_frequency) if freq>1000]
   print(len(most_active))


   fig = plt.figure(1)
   plt.hist(activity_frequency, log=True)
   plt.show()

   for unit in most_active:
       enr.get_print_unit_biology(unit, os.path.join('results', 'biology_result.txt'))
