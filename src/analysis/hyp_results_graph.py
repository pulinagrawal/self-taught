import os
import pandas as pd
import matplotlib as plt

to_graph_file = os.path.join('results', 'to_graph.csv')
data = pd.read_csv(to_graph_file, sep=',')

data.