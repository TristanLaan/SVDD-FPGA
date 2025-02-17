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


#####  Import the model.py file #####
home = os.getcwd() + '/..'
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
logger = logging.getLogger('svdd-hls4ml')
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

#################
#plotting
import matplotlib.pyplot as plt


training_filename = home + '/data/training.h5'
testing_filename = home + '/data/testing.h5'

def main(Flags):
    hl_int_list, model_name = utils.getModelName(Flags)

    tf.keras.backend.set_floatx(Flags.precision)
    print("using %s precision" % (tf.keras.backend.floatx()))

    training_data, testing_data,regression_training,regression_testing,etype_testing,etype_training = utils.preprocessdata(training_filename,testing_filename,Flags)
    modelpath = os.path.join(home,Flags.modeldir,"savedmodels",model_name + '.h5')

    logger.info("starting inference")
    if mode == 'ordered':
        data_dim = regression_testing.shape[1]
        dataset_len = regression_testing.shape[0]
    utils.test(dataset_len, data_dim, etype_testing, training_data, testing_data, model_name, Flags,modelpath=modelpath)
    logger.info("finished inference")



if __name__ == '__main__':
    from utils import SVDD_arguments_parser

    FLAGS, unparsed = SVDD_arguments_parser()

    #os.environ['PATH'] = '/opt/xilinx/tools/Vivado/2022.2/bin/:' + os.environ['PATH']
    print(os.environ['PATH'])
    main(FLAGS)
