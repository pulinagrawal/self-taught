import numpy as np
import pickle as pkl
import os
from utils import ae
import matplotlib.pyplot as plt

def reshape_for_display(image_pixels):
    shape = np.shape(image_pixels)
    if (shape[1] ** 0.5) - int(shape[1] ** 0.5) == 0:
        image = np.reshape(image_pixels, [shape[0], int(shape[1] ** 0.5), int(shape[1] ** 0.5)])
    else:
        image = np.reshape(image_pixels, [shape[0], shape[1], 1])

    return image

def plot_genes(input_genes1, input_genes2=None, title=''):
    fig = plt.figure(1)
    if input_genes2 is not None:
        plt.subplot(211)
    img = plt.imshow(input_genes1)
    plt.title(title)
    if input_genes2 is None:
        fig.colorbar(img, shrink=0.5, aspect=5)
        plt.ylabel('Samples')
        plt.xlabel('Genes')
    if input_genes2 is not None:
        plt.subplot(212)
        img = plt.imshow(input_genes2)
    plt.show()

    return fig

if __name__ == '__main__':

    data = pkl.load(open(os.path.join('data', 'normd_split_1.pkl'), 'rb'))
    sae = ae.Autoencoder.load_model(os.path.join('results', 'best_attmpt_2', 'geodb_ae_89.net'))
    input_image = data[2].next_batch(100)
    print(input_image[1])
    print(sae.rho)
    print(sae.beta)
    print(np.shape(input_image[0]))
    output_image = sae.reconstruct(input_image[0])
    print(np.shape(output_image))
    plot_genes(input_image[0], output_image)
    #img = plt.imshow(output_image)

