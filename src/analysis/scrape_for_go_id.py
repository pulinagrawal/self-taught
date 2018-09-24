import csv
import math
import time
import sys
import itertools
import re
import tqdm
import rpy2
import rpy2.robjects as ro
import numpy as np
from collections import defaultdict
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

def get_result_file_dict(biology_result_file):
    biology = defaultdict(list)
    with open(biology_result_file) as f:
        for line in f:
            if 'unit' in line:
                if '_gsa' in line:
                    unit = int(re.findall(r'\d+', line)[-1])
                    while True:
                        line = f.readline()
                        geneset = line.split('\t')[0]
                        if geneset != '\n' and geneset != '':
                            biology[unit].append(geneset)
                        if line == '\n' or line == '':
                            break
    return biology

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)

def get_go_id(go_term):
    time.sleep(1)
    raw_html = simple_get('http://software.broadinstitute.org/gsea/msigdb/cards/{0}.html'.format(go_term))
    html = BeautifulSoup(raw_html, 'html.parser')
    a = html.find_all("th", string='Exact source')
    go_id = list(a[0].parent.children)[3].string
    return go_id

def map_from_file(go_terms_map_file):
    term_map = {}
    file = csv.reader(open(go_terms_map_file))
    for row in file:
        term_map[row[0]] = row[1]
    return term_map

def build_map(go_terms_filename, term_map_filename='', from_web=False, write_updated_file=True):
    file_term_map = {}
    if term_map_filename != '':
       file_term_map = map_from_file(term_map_filename)
    term_map = {}
    file = csv.reader(open(go_terms_filename))
    for row in file:
        go_term = row[0]
        if go_term in file_term_map:
            term_map[go_term] = file_term_map[go_term]
        else:
            if from_web:
                term_map[go_term] = get_go_id(go_term)
                print('retrieved go_id of {0} from web'.format(term_map[go_term]))

    if write_updated_file and from_web:
        if term_map_filename == '':
            term_map_filename = 'go_terms_for_'+go_terms_filename+'_.txt'
        else:
            term_map_filename = term_map_filename.split('.')[0]+'_updated_from_web.txt'

        with open(term_map_filename, 'w') as term_map_file:
            csv_file = csv.writer(term_map_file)
            for go_term in term_map:
                csv_file.writerow((go_term, term_map[go_term]))

    return term_map

