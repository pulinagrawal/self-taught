import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.utils import ae
from src.utils import image as imtools
import numpy as np

filename = os.path.join('results', 'best_attmpt_2', 'gedb_ae_89.net')

auto = ae.Autoencoder.load_model(filename)
feature_weights = np.transpose(auto.weights)
fig = plt.figure(figsize=(10, 10))
feature_images = imtools.reshape_for_display(feature_weights)
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    ax.imshow(feature_images[i], cmap=cm.gray)

save_path = str(filename).split('.')[-2]+'_features.png'
print(save_path)
fig.savefig(save_path)