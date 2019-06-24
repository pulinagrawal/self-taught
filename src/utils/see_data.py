import os
import sys
import pandas as pd
import pickle as pkl
from src.utils import image as im


if __name__ == '__main__':

    pickled = True
    create_sets = True
    normed = False
    if len(sys.argv) > 2:
        filename = sys.argv[1]
    else:
        filename = os.path.join(os.path.pardir, os.path.pardir, 'data', 'final_transp_directpkl.pkl')

    if os.path.splitext(filename)[1] == '.txt':
        iter_csv = pd.read_csv(filename, sep='\t', index_col=0, chunksize=20000)
        df = pd.concat([chunk for chunk in iter_csv])
    else:
        df = pkl.load(open(filename, 'rb'))

    fig = im.plot_genes(df.sample(1000))
    fig.savefig(os.path.splitext(filename)[0]+'.png')

