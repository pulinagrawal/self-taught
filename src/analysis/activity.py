import os
import matplotlib.pyplot as plt
import numpy as np
from src.analysis import setup_analysis, get_activations, build_genesetlist_from_units

def main():
    model_name = 'geodb_ae_89.net'
    model_folder = os.path.join('results', 'best_attmpt_2')

    labelled_data_files = ['GSE15061_aml.txt', 'GSE15061_mds.txt']

    result_filename = 'best_sparse_biology.txt'

    normed_split_path = os.path.join('data', 'normd_split_')
    split = 1

    for_top_x_pct_units = 0.02

    datasets, geneset_unit_map, gsm_groups, model = setup_analysis(labelled_data_files, model_folder, model_name,
                                                               normed_split_path, result_filename, split)
    genesets_sets = []
    for class_id in gsm_groups:
        all_activations = []
        i=0
        for gsm in gsm_groups[class_id]:
            activation = get_activations(gsm, datasets, model)
            all_activations.append(activation)
            i += 1
        all_activations = np.array(all_activations)
        mean_activations = all_activations.mean(axis=0)
        print(mean_activations)
        fig = plt.figure()
        box_values = all_activations.transpose()[:50]
        plt.boxplot(box_values.transpose())
        plt.xlabel('Unit')
        plt.title(labelled_data_files[class_id])
        fig.show()
        fig = plt.figure()
        plt.hist(mean_activations)
        plt.xlabel('Mean Activation of a Unit')
        plt.title(labelled_data_files[class_id])
        fig.show()
        fig = plt.figure()
        box_values = all_activations[:50]
        plt.boxplot(box_values.transpose())
        plt.xlabel('Sample')
        plt.title(labelled_data_files[class_id])
        fig.show()
        mean_of_means = mean_activations.mean()
        std_of_means = mean_activations.std()
        print(mean_of_means)
        print(std_of_means)
        most_active_units = [i for i, activation in enumerate(mean_activations) if activation>0.125]
        genesets_sets.append(set(build_genesetlist_from_units(most_active_units, geneset_unit_map)))
        print(len(all_activations.mean(axis=0)))

    print('Common', genesets_sets[0].intersection(genesets_sets[1]))
    print()
    print('AML', genesets_sets[0] - genesets_sets[1])
    print()
    print('MDS', genesets_sets[1] - genesets_sets[0])

main()

