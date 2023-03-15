#!/usr/bin/env python
# coding: utf-8

################################################################################################################
################################################################################################################
import logging
# create logger
logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
import os.path
home = os.getcwd()

# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
################################################################################################################
################################################################################################################

#####  Tensorflow related #####
import tensorflow as tf
from sklearn.utils import shuffle
import os
import numpy as np
import glob
from tqdm import tqdm
import sys, argparse
    
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler 


import csv
import pandas as pd

mode = 'ordered'
import modeldefault
import dataloader
import plotting

from sklearn.metrics import roc_auc_score
from sklearn import metrics

xcoord = ['Ato4l', 'hChToTauNu', 'hToTauTau', 'leptoquark']

"""
Ato4l : 1
background : 2
hChToTauNu : 3
hToTauTau : 4
leptoquark : 5
"""

sgn_dict = {'Ato4l': 1,'SM': 2, 'hChToTauNu': 3, 'hToTauTau': 4, 'leptoquark': 5}



def SVDD_arguments_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=10000, help='Mini batch size')
    
    parser.add_argument('--dim', type=int, default=55, help='Latent space dim.')
    
    parser.add_argument('--ap_fixed_width', type=int, default=32, help='ap_fixed<X,X>')

    parser.add_argument('--ap_fixed_int', type=int, default=6, help='ap_fixed<X,X>')

    parser.add_argument('--hidden_layers', type=str, default="512 256 128", help='Number of nodes in hidden layers')   

    parser.add_argument('--device', type=str, default='cpu', help='type of device cpu or gpu')

    parser.add_argument('--plotdata', type=bool, default=False, help='plot the testing data')

    parser.add_argument('--plotdir', type=str, default="plots", help='plotdir')

    parser.add_argument('--quantised', type=bool, default=False, help='quantised')

    parser.add_argument('--modeldir', type=str, default="models_trained", help='model dir')

    parser.add_argument('--sparsity', type=str, default="", help='sparsity')

    parser.add_argument('--inmodeldir', type=str, default="", help='in model dir')

    parser.add_argument('--resume', type=bool, default=False, help='Resume')

    parser.add_argument('--hls4ml', type=bool, default=False, help='put quantised model in hls4ml wrapper')

    parser.add_argument('--train', type=bool, default=False, help='Resume')
    
    parser.add_argument('--test', type=bool, default=True, help='Resume')

    parser.add_argument('--run', type=bool, default=False, help='Run')

    parser.add_argument('--fixed_target', type=int, default=1)

    parser.add_argument('--mode', type=str, default='ordered')

    parser.add_argument('--build', type=bool, default=False, help='build model')

    parser.add_argument('--profiling', type=bool, default=False, help='profiling model')

    parser.add_argument('--iterations', type=int, default=1, help="number of times the inference procedure is repeated")

    parser.add_argument('--precision', type=str, default="float32", help="set precision (default = float32)")

    parser.add_argument('--maxEvents', type=int, default=-1, help="Set the number of events to run over, -1 is all events")

    parser.add_argument('--epochs', type=int, default=1, help="Set the number of epochs")

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

def getModelName(Flags):

    dim_z = Flags.dim 
    ft = Flags.fixed_target
    map_object = map(int, Flags.hidden_layers.split())
    hl_int_list = list(map_object)
    #### generate name from FLAGS ######
    if mode == 'ordered':
      model_name = 'SVDD_{0}l'.format(len(hl_int_list))
      for i in hl_int_list:
        model_name += '_{0}'.format(i)
      model_name += '_bs_10000'  + '_' + mode + "_ft_%i_zdim_%i" % (ft,dim_z)
    logger.info('starting run of model: %s ' %(model_name))

    return hl_int_list, model_name


def preprocessdata(training_filename,testing_filename,Flags):
    h5_fname = 'h5_ordered'
    if mode == 'ordered':
        etype_training, regression_training = dataloader.unpack_ordered(training_filename, Flags,Flags.precision)
    if Flags.plotdata:
        plot(etype_training,regression_training,Flags)
    #load training data
    scaler = StandardScaler()
    scaler.fit(regression_training)
    reg_normalized_training = scaler.transform(regression_training)
    if mode == 'ordered':
        training_data = reg_normalized_training[:]


    if mode == 'ordered':
        etype_testing, regression_testing = dataloader.unpack_ordered(testing_filename, Flags,Flags.precision)

    reg_normalized_testing = scaler.transform(regression_testing)

    if mode == 'ordered':
        testing_data = reg_normalized_testing[:]

    return training_data, testing_data,regression_training,regression_testing,etype_testing,etype_training



def train(dataset_len, data_dim, training_data,model_name, Flags):
    mode = Flags.mode
    dim_z = Flags.dim 
    ft = Flags.fixed_target
    map_object = map(int, Flags.hidden_layers.split())
    hl_int_list = list(map_object)
    modeldir= Flags.modeldir

    print("INFO:: loading model :")
    print("test data shape")
    print(training_data.shape)
    print("training_data shape")
    print(data_dim)
    print("dim_z")
    print(dim_z)
    print("ft")
    print(ft)
    print("model_name")
    print(model_name)

    logger.info('training ft= %s ,dimz =  %s' %(int(ft),int(dim_z) ) )
    with tf.device("/cpu:0"):
        #sess = tf.Session()
        #K.set_session(sess)
        sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)

        model = modeldefault.VariationalAutoencoderModel(hl_int_list,model_name, data_dim, dataset_len, int(dim_z), int(ft), mode=mode, verbose=True,modeldir=Flags.modeldir,quantised=Flags.quantised,ap_fixed_width=Flags._width,ap_fixed_int=Flags.ap_fixed_int, Flags = Flags)
        model.train_model(training_data,epochs=Flags.epochs)


    return model


def test(dataset_len, data_dim, etype_testing, training_data, testing_data, model_name, Flags,model=False,modelpath=False):
    #### READ FLAGS ######
    dim_z = Flags.dim 
    ft = Flags.fixed_target
    map_object = map(int, Flags.hidden_layers.split())
    hl_int_list = list(map_object)
    modeldir=Flags.modeldir
    print("INFO:: loading model:")
    print("test data shape")
    print(testing_data.shape)
    print("test data shape")
    print(data_dim)
    print("dim_z")
    print(dim_z)
    print("ft")
    print(ft)
    print("model_name")
    print(model_name)

    with tf.device("/{0}".format(Flags.device)):
        sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)

        if not model:
            #create empty SVDD model and load weights
            if modelpath:
                model = modeldefault.VariationalAutoencoderModel(hl_int_list, model_name, data_dim, dataset_len, int(dim_z), int(ft), mode=mode, verbose=True,quantised=Flags.quantised,hls4ml=Flags.hls4ml,modeldir=Flags.modeldir,modelpath=modelpath,ap_fixed_width=Flags.ap_fixed_width,ap_fixed_int=Flags.ap_fixed_int,Flags = Flags)
            else:
                model = modeldefault.VariationalAutoencoderModel(hl_int_list, model_name, data_dim, dataset_len, int(dim_z), int(ft), mode=mode, verbose=True,quantised=Flags.quantised,hls4ml=Flags.hls4ml,modeldir=Flags.modeldir,ap_fixed_width=Flags.ap_fixed_width,ap_fixed_int=Flags.ap_fixed_int,Flags= Flags)

        #Evaluate radius for training with output r_max
        training_data = shuffle(training_data)
        testing_data, etype_testing = shuffle(testing_data, etype_testing)

        if Flags.maxEvents > 0:
            training_data   = training_data[:Flags.maxEvents]
            testing_data    = testing_data[:Flags.maxEvents]
            etype_testing   = etype_testing[:Flags.maxEvents]



        r_max = model.evaluate_radius_max(training_data,100000)
        logger.debug("radius max is: %s" %(r_max))

        if r_max == 0.0:
            r_max = 1.e-35

        #evaluate output vector of the SVDD and calculate the scores
        # for i in range(0, Flags.iterations, 1):
        
        scores = model.evaluate_radius(testing_data, r_max,Flags.batch)

        # """
        # Ato4l : 1
        # background : 2
        # hChToTauNu : 3
        # hToTauTau : 4
        # leptoquark : 5
        # """

        mask_bkg = (etype_testing == 2)
        logger.debug("mask_bkg.shape: %s" %(mask_bkg.shape))
        logger.debug("scores.shape: %s" %(scores.shape))



        # loop over the event types to get the score from all of them
        for key in sgn_dict:
            logger.debug("signal key: %s" %(key))

            AUC = []
            Epsilon1 = []
            Epsilon2 = []
            Epsilon3 = []
            Epsilon4 = []
            mask_sgn = (etype_testing == sgn_dict[key])

            y_true = np.concatenate (
                (np.zeros(scores[mask_bkg].shape[0], dtype="float32"),  ############ lake length of bkg samples and create len zeros
                                     np.ones(scores[mask_sgn].shape[0], dtype="float32")) ############ lake length of sig samples and create len zeros
                                    )


            logger.debug("preview output:")
            print(scores[mask_sgn])
            print(scores[mask_bkg])

            y_test = np.concatenate((scores[mask_bkg], scores[mask_sgn]))
            logger.debug("calculating ROC score")
            auc = roc_auc_score(y_true, y_test)

            logger.debug("calculating thresholds")
            print(y_true)
            print(y_test)

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_test)
            eps1, eps2, eps3, eps4 = modeldefault.evaluate_efficiencies(y_true, y_test)

            AUC = []
            AUC.append(auc)
            Epsilon1.append(eps1)
            Epsilon2.append(eps2)
            Epsilon3.append(eps3)
            Epsilon4.append(eps4)


            # if Flags.hls4ml:
            #     outdirscores = os.path.join('results',Flags.modeldir,"hls4ml_wrapper",'scores',model_name,key)
            #     outdirROCs = os.path.join('results',Flags.modeldir,"hls4ml_wrapper",'ROCs',model_name,key)
            #     outdirsmetrics = os.path.join('results',Flags.modeldir,"hls4ml_wrapper",'metrics',model_name,key)
            # else:





            modeldir = Flags.modeldir + "_" + str(Flags.ap_fixed_width)+"_"+ str(Flags.ap_fixed_int)
            
            outdirscores = os.path.join('results',modeldir,'scores',model_name,key)
            outdirROCs = os.path.join('results',modeldir,'ROCs',model_name,key)
            outdirsmetrics = os.path.join('results',modeldir,'metrics',model_name,key)


            if not os.path.exists(outdirscores):
                os.makedirs(outdirscores)
            if not os.path.exists(outdirROCs):
                os.makedirs(outdirROCs)
            if not os.path.exists(outdirsmetrics):
                os.makedirs(outdirsmetrics)
            logger.debug("saving scores in: %s" %(outdirscores))
            logger.debug("saving outdirsmetrics in: %s" %(outdirsmetrics))

            with open(os.path.join(outdirROCs,'fpr.txt'),"w") as f:
                np.savetxt(f,fpr)
            with open(os.path.join(outdirROCs,'tpr.txt'),"w") as f:
                np.savetxt(f,tpr)
            # Save scores individually
            with open(os.path.join(outdirscores,'signal_scores.txt'),"w") as f:
                np.savetxt(f,scores[mask_sgn])
            with open(os.path.join(outdirsmetrics,'AUC.txt'),"w") as f:
                np.savetxt(f,AUC)
            with open(os.path.join(outdirsmetrics,'Epsilon1.txt') ,"w") as f:
                np.savetxt(f,Epsilon1)
            with open(os.path.join(outdirsmetrics,'Epsilon2.txt') ,"w") as f:
                np.savetxt(f,Epsilon2)
            with open(os.path.join(outdirsmetrics,'Epsilon3.txt') ,"w") as f:
                np.savetxt(f,Epsilon3)
            with open(os.path.join(outdirsmetrics,'Epsilon4.txt') ,"w") as f:
                np.savetxt(f,Epsilon4)