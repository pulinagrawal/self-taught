import csv
import time
import sys
from . import get_result_file_dict
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

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

def build_map(go_terms_file, term_map_file=''):
    file_term_map = {}
    if term_map_file != '':
       file_term_map = map_from_file(term_map_file)
    term_map = {}
    file = csv.reader(open(go_terms_file))
    for go_term in file:
        if go_term in file_term_map:
            term_map[go_term] = file_term_map[go_term]
        else:
            term_map[go_term] = get_go_id(go_term)
    return term_map

if __name__ == "__main__":
    biology_filename = sys.argv[1]
    go_terms_filename = sys.argv[2]
    try:
        term_map_filename = sys.argv[3]
    except IndexError:
       term_map_filename = ''
    term_map = build_map(go_terms_filename, term_map_filename)
    biology_map = get_result_file_dict(biology_filename)

