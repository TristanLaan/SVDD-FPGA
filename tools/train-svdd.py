#####  Python essentials #####
import numpy as np
import glob
from tqdm import tqdm
import sys
import re
import math
import itertools
from pathlib import Path
import argparse
import time
import os.path
import pprint



#####  Tensorflow related #####
import tensorflow as tf
sgn_dict = {'Ato4l': 1, 'hChToTauNu': 3, 'hToTauTau': 4, 'leptoquark': 5}
mode = 'ordered'

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function



#####  Import the model.py file #####
home = os.getcwd()
sys.path.append(os.path.join(home,"code"))
import modeldefault
import dataloader
import utils

from sklearn.metrics import roc_auc_score
from sklearn import metrics




################################################################################################################
################################################################################################################
import logging
# create logger
logger = logging.getLogger('train-svdd')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

# 'application' code
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warn message')
# logger.error('error message')
# logger.critical('critical message')
################################################################################################################
################################################################################################################
# #####  Tensorflow related #####
# import tensorflow as tf
# from keras.models import Model
# from keras.losses import mse, binary_crossentropy, categorical_crossentropy
# from keras import backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# from keras import regularizers
# from scipy.stats import multivariate_normal
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from sklearn.metrics import log_loss, roc_curve
# from sklearn.neighbors import KernelDensity, DistanceMetric


# #####  Unknown where used #####
# import h5py
# from multiprocessing.dummy import Pool as ThreadPool
# import subprocess
# import random
# from joblib import Parallel, delayed
# import multiprocessing
# import ast


#################
#plotting
import matplotlib.pyplot as plt




training_filename = home + '/data/training.h5' 
testing_filename = home + '/data/testing.h5' 

def main(Flags):


    hl_int_list, model_name = utils.getModelName(Flags)

    tf.keras.backend.set_floatx(Flags.precision)
    print("using %s precision" % (tf.keras.backend.floatx()))

    #create folders to store results
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('results/scores/'):
        os.makedirs('results/scores/')

    if not os.path.exists('results/metrics'):
        os.makedirs('results/metrics')

    training_data, testing_data,regression_training,regression_testing,etype_testing,etype_training = utils.preprocessdata(training_filename,testing_filename,Flags)

    if Flags.train:
        if mode == 'ordered':
            data_dim = regression_training.shape[1]
            dataset_len = regression_training.shape[0]
        model = utils.train(dataset_len, data_dim, training_data,model_name, Flags)
        logger.info("finished training")

    if Flags.run:
        logger.info("starting inference")
        if mode == 'ordered':
            data_dim = regression_testing.shape[1]
            dataset_len = regression_testing.shape[0]
        utils.test(dataset_len, data_dim, etype_testing, training_data, testing_data, model_name, Flags,model=model)
        logger.info("finished inference")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=10000, help='Mini batch size')
    
    parser.add_argument('--dim', type=int, default=55, help='Latent space dim.')

    parser.add_argument('--hidden_layers', type=str, default='8', help='Number of nodes in hidden layers')   
   
    parser.add_argument('--device', type=str, default='cpu', help='type of device cpu or gpu')

    parser.add_argument('--plotdata', type=bool, default=False, help='plot the testing data')

    parser.add_argument('--plotdir', type=str, default="plots", help='plotdir')

    parser.add_argument('--quantised', type=bool, default=False, help='quantised')

    parser.add_argument('--modeldir', type=str, default="models_trained", help='model dir')

    parser.add_argument('--resume', type=str, default='False', help='Resume')

    parser.add_argument('--train', type=str, default='False', help='Resume')
    
    parser.add_argument('--test', type=str, default='True', help='Resume')

    parser.add_argument('--run', type=bool, default=False, help='Run')

    parser.add_argument('--fixed_target', type=int, default=1)

    parser.add_argument('--mode', type=str, default='ordered')

    parser.add_argument('--iterations', type=int, default=1, help="number of times the inference procedure is repeated")

    parser.add_argument('--precision', type=str, default="float32", help="set precision (default = float32)")

    FLAGS, unparsed = parser.parse_known_args()
   
    main(FLAGS)
