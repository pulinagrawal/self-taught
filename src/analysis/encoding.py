from src.utils import ae
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
    # geodb_training used 'transp_normd_1norm.pkl', does not matter, no genes removed between the steps
    return dataset.columns

if __name__ == '__main__':

   model_name = 'geodb_ae_91.net'
   model_folder = os.path.join('results', 'best_dense_attmpt_1')
   model_file = os.path.join(model_folder, model_name)

   model = ae.Autoencoder.load_model(model_file, logdir=os.path.join('results', 'features_'+model_name))
   features = pd.DataFrame.from_records(model.weights[0])
   fig = plt.figure(1)
   plt.hist(features[0])
   print(np.std(features[0]))
   plt.show()
   output_file = os.path.join(model_folder, 'features_'+model_name.split('.')[0]+'.csv')

   genelist = get_gene_list()

   features = features.set_index(genelist)

   for idx in features.iterrows():
       geneids = str(idx[0]).split(' /// ')
       if len(geneids) > 1:
           features = features.drop(idx[0])


   print(features.shape)
   features.to_csv(output_file, sep='\t')
