import datetime as dt
from collections import defaultdict
import os
from functools import lru_cache


def results_timestamp_dir():
    timestamp = str(dt.datetime.now())
    timestamp = timestamp.replace(' ', '_').replace(':', '-').replace('.', '-')
    project_path = os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0]
    _run_folder = os.path.join(project_path, 'results', timestamp)
    return _run_folder


def reverse_dict_of_lists(dict_of_lists):
    reverse_dict = defaultdict(list)
    for k in dict_of_lists:
        for list_item in dict_of_lists[k]:
            reverse_dict[list_item].append(k)
    return reverse_dict


def get_gsm_labels(file_list, folder='.'):
    gsm_labels = {}
    i = 0
    for file in file_list:
        with open(os.path.join(folder, file)) as f:
            header = f.readline()
            header = header.split('\n')[0]
            gsms = header.split('\t')[1:]
            gsm_labels[file] = gsms
            i = i+1
    return gsm_labels

