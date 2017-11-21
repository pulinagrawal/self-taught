from src.utils import ae
from collections import Counter
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

def get_input(gsm, datasets):
    for dataset in datasets:
        for i, label in enumerate(dataset.labels):
            if label == gsm:
                break
        return dataset.images[i]
    return None

def get_print_unit_biology(unit_num, biology_result_file):
    biology = []
    with open(biology_result_file) as f:
       for line in f:
           if 'unit'+str(unit_num)+'_gsa' in line:
               print('unit '+str(unit_num))
               while True:
                   line = f.readline()
                   geneset = line.split('\t')[0]
                   if geneset != '\n':
                       biology.append(geneset)
                   print(line, end='')
                   if line == '\n':
                       break
               break
    return biology

def get_gsm_biology(gsm, biology_result_file, model, datasets, for_top_x_pct_units=0.02):

   input_genes = get_input(gsm, datasets)
   if input_genes is not None:
       model_input = np.expand_dims(input_genes, axis=0)
       encoding = model.encoding(model_input)

   sorted_units = sorted(enumerate(encoding[0]), key=lambda unit: unit[1])
   top_units = [unit+1 for unit, _ in sorted_units[-int(len(sorted_units) * for_top_x_pct_units):]]
   #added one to unit number because biology_results file assume units start with 1

   biology = []
   for unit in top_units:
       biology.extend(get_print_unit_biology(unit, biology_result_file))

   return biology


if __name__ == '__main__':

   model_name = 'geodb_ae_91.net'
   model_file = os.path.join('results', 'best_dense_attmpt_1', model_name)
   gsm_list = ['GSM2376031','GSM2376032','GSM2376033','GSM2376034']
   result_filename = 'dense_biology_results.txt'
   bio_result_file = os.path.join('results', result_filename)

   model = ae.Autoencoder.load_model(model_file, logdir=os.path.join('results', 'features_'+model_name))
   normed_split_path = os.path.join('data', 'normd_split_')
   split = 1
   unlabelled, labelled, validation, test = pkl.load(open(normed_split_path+str(split)+'.pkl', 'rb'))

   gsm_list_biology = []
   for gsm in gsm_list:
       gsm_list_biology.extend(get_gsm_biology(gsm, bio_result_file, model, [unlabelled, validation]))

   for geneset, freq in sorted(Counter(gsm_list_biology).items(), key=lambda x: x[1], reverse=True):
       print(geneset,': ',freq)
