import os
import argparse
import numpy as np
import pickle as pkl

from utils import ae
from utils import ffd
import utils
import trainer
from utils import image as im
#from src.utils import image as im

def get_gsm_list(filename):
    gsmlist = []
    with open(filename, 'rb') as gsmlistfile:
        for row in gsmlistfile:
            gsmlist.append(row)

if __name__ == '__main__':

    stacked = False
    normed_split_path = os.path.join('data', 'scaled_shftd_split_')
    parser = argparse.ArgumentParser(description='run on hyperparameters')
    parser.add_argument('--nHidden', type=int, help='number of units in hidden layer')
    parser.add_argument('--learning_rate', type=float, help='learning rate for unsupervised step')
    parser.add_argument('--sparsity', type=float, help='sparsity of the hidden layer')
    parser.add_argument('--beta', type=float, help='value of beta to control the weight of sparsity constraint in loss')
    parser.add_argument('--keep_prob', type=float, default=1, help='probability to keep a hidden unit during training (implements dropout)')
    parser.add_argument('--denoise_keep_prob', type=float, default=1, help='probability to keep a input during training (possibly implements denoising)')
    parser.add_argument('--lambda_', type=float, default=0, help='multiplier for weight decay parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--multiplier', type=float, default=10, help='gradient descent momentum')
    parser.add_argument('--split', type=int, help='the data split index to use')
    args = parser.parse_args()
    
    #unlabelled, labelled, validation, test = create_datasets(os.path.join(os.pardir, 'data', 'final_transp_directpkl.pkl'))
    #datasets = create_datasets(os.path.join(os.pardir, 'data', 'tranposed_direct.pkl'))
    #pkl.dump(datasets, open('..\\data\\datasets_dropped.pkl', 'wb'))
    # Loading from splits is needed because of different version of pandas on hpc

    unlabelled, labelled, validation, test = pkl.load(open(normed_split_path+str(args.split)+'.pkl', 'rb'))
    input_size = unlabelled._images.shape[1]

    if stacked:
        net1 = ae.Autoencoder.load_model(os.path.join('results', 'best_attmpt_2', 'geodb_ae_89.net'))
        unlabelled._images = net1.encoding(unlabelled.next_batch(unlabelled.num_examples)[0])
        validation._images = net1.encoding(validation.next_batch(validation.num_examples)[0])
        input_size = net1._network_architecture[-1]

    print(args)

    logdir = utils.results_timestamp_dir()
    '''
    from deepautoencoder import StackedAutoEncoder as SAE
    model = SAE(dims=[args.nHidden], loss='rmse', activations=['sigmoid'], epoch=[10000], batch_size=128, lr=args.learning_rate, print_step=200)
    result = model.fit(unlabelled.next_batch(unlabelled.num_examples)[0])
    '''
    geo_trainer = trainer.SelfTaughtTrainer(ae.Autoencoder([input_size, args.nHidden],
                                                           learning_rate=args.learning_rate,
                                                           sparsity=args.sparsity,
                                                           beta=args.beta,
                                                           logdir=logdir,
                                                           lambda_=args.lambda_,
                                                           momentum=args.momentum,
                                                           keep_prob=args.keep_prob,
                                                           tf_multiplier=args.multiplier,
                                                           denoise_keep_prob=args.denoise_keep_prob
                                                           ),
                                            ffd.FeedForwardNetwork([args.nHidden, 10],),
                                            batch_size=128,
                                            unlabelled=unlabelled,
                                            labelled=labelled,
                                            validation=validation,
                                            test=test,
                                            save_filename='geodb_lvl2',
                                            run_folder=logdir
                                            )

    geo_trainer.run_unsupervised_training()
    print(logdir)
