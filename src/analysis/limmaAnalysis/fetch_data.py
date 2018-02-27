from src.analysis import enrichment as enr
from src import utils
import pickle as pkl
import pandas as pd
import os

def get_gene_list(filename):
    genes = []
    with open(filename) as f:
        for line in f:
            if str.isdigit(line[0]):
                genes.append(line.split('\t')[0])
    return tuple(genes)

data_files = ['GSE8052_asthma_0.txt', 'GSE8052_asthma_1.txt']
data_folder = os.path.join('data')

split = 1
normed_split_path = os.path.join('data', 'scaled_split_')
unlabelled, labelled, validation, test = pkl.load(open(normed_split_path + str(split) + '.pkl', 'rb'))

gsm_labels = utils.get_gsm_labels(data_files, data_folder)

genelist = get_gene_list(os.path.join('data', 'entrez_genes.txt'))

filename = os.path.join('data', 'final_transp_directpkl.pkl')
df = pkl.load(open(filename, 'rb'))

data = pd.DataFrame(index=df.columns,
                    columns=[gsm for class_id in gsm_labels for gsm in gsm_labels[class_id]],
                    dtype=float)

for class_id in gsm_labels:
    for gsm in gsm_labels[class_id]:
        data[gsm] = enr.get_input(gsm, [unlabelled, validation])

limma_filename = 'extracted_'+data_files[0].split('_')[0]+'.csv'
data.to_csv(os.path.join('data', limma_filename), sep='\t')
design = pd.DataFrame.from_dict(utils.reverse_dict_of_lists(gsm_labels))
design = design.T
design[1] = [1]*len(design.index)
design.to_csv(os.path.join('data', 'design_'+limma_filename), sep='\t')

