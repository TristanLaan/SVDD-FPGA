
import numpy as np
import math
import sys
import tensorflow as tf

from tensorflow.keras.layers import Lambda, Input, Dense, LeakyReLU, BatchNormalization, Concatenate, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import regularizers
from scipy.stats import multivariate_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

from sklearn.metrics import log_loss, roc_curve

from glob import glob
from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import mean_squared_error, roc_auc_score
import importlib
import h5py
import hls4ml
import plotting




########################


########################

################################################################################################################
################################################################################################################
import logging
# create logger
logger = logging.getLogger('modeldefault')
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


def get_R(coords, center=None):
    return np.linalg.norm(coords,axis=1)


class VariationalAutoencoderModel():
    def __init__(self, hidden_layers, filename, D, dataset_len, dim_z, c, mode=None, verbose=False,modeldir="models",quantised=False,hls4ml=False,modelpath=False):
        self.D = D
        self.dataset_len = dataset_len
        self.dim_z = dim_z
        self.c = c
        self.verbose = verbose
        self.hls4ml_model = None

        self.model_filename = str(modeldir)+ '/' + filename + '.h5'
        self.modeldir = str(modeldir)
        self.modelpath = modelpath
        self.use_hls4ml = hls4ml
        self.filename = filename
        self.mode = mode
        self.hidden_layers = hidden_layers

        # if hls4ml and quantised:

        if modelpath:
            if "models_conventional" in self.modeldir:
                logger.info("Building NORMAL model")
                model = self.build_lhcdata_model()
                model.load_weights(self.modelpath)

            elif quantised:
                logger.debug('Loading quantised model: %s' %(modelpath))
                model = self.load_lhcdata_model_quantized()
            else:
                logger.debug('Loading standard model: %s' %(modelpath))
                model = self.load_lhcdata_model_quantized()
                model.summary()

        elif quantised:
            logger.info("Building QUANTIZED model")
            model = self.build_lhcdata_model_quantised()
        else:
            logger.info("Building NORMAL model")
            model = self.build_lhcdata_model()

        if self.verbose:
            model.summary()

        self.model = model
        return
    
    def z_log_var_activation(self, x):
        return K.sigmoid(x) * 10

    def MSE(self, c, dataset_len, dim_z):
        output_data = np.zeros((dataset_len, dim_z), np.int8)
        logger.debug("MSE:")
        output_data.fill(c)
        print(output_data)
        return output_data

    def load_lhcdata_model_quantized(self):

        from qkeras.utils import _add_supported_quantized_objects
        co = {}
        _add_supported_quantized_objects(co)
        model = load_model(self.modelpath, custom_objects=co)
        model.summary()

        
        # instantiate encoder model
        if self.mode == 'ordered':
            self.encoder = Model(model.input, model.layers[-1].output, name='encoder')
        self.encoder.compile(optimizer='adam', loss='mean_squared_error')


        return self.encoder

    def load_lhcdata_model(self):
        model = load_model(self.modelpath)
        logger.debug('loaded model with summary: %s' %(self.modelpath))
        model.summary()


        if self.mode == 'ordered':
            self.encoder = Model(model.input, model.layers[-1].output, name='encoder')
        self.encoder.compile(optimizer='adam', loss='mean_squared_error')


        return self.encoder

    def build_lhcdata_model(self):
        D = self.D
        if self.mode == 'ordered':
            in_regression = Input(shape=(D,), name='in_regression')
            inputs = in_regression

        x = Dense(self.hidden_layers[0], activation='elu')(inputs)
        if len(self.hidden_layers) > 1:
            for i, v in enumerate(self.hidden_layers):
                if (i > 0):
                    x = Dense(v, activation='elu')(x)
        """
        x = Dense(512, activation='elu')(inputs)
        x = Dense(256, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        #
        x = Dense(256, activation='elu')(inputs)
        x = Dense(128, activation='elu')(x)
        
        x = Dense(8, activation='elu')(inputs)
        """
        z_mean = Dense(self.dim_z, name='z_mean', activation='linear')(x)
        # instantiate encoder model
        if self.mode == 'ordered':
            self.encoder = Model([in_regression], [z_mean], name='encoder')

        self.encoder.compile(optimizer='adam', loss='mean_squared_error')
        return self.encoder


    def build_lhcdata_model_quantised(self):
        D = self.D

        if self.mode == 'ordered':
            in_regression = Input(shape=(D,), name='in_regression')
            inputs = in_regression

        x = QDense( self.hidden_layers[0],
                    kernel_quantizer=quantized_bits(8,0,alpha=1), 
                    bias_quantizer=quantized_bits(8,0,alpha=1),
                    activation=quantized_relu(8)      )(inputs)


        if len(self.hidden_layers) > 1:
            for i, v in enumerate(self.hidden_layers):
                if (i > 0):
                    x = QDense(v,kernel_quantizer=quantized_bits(8,0,alpha=1), 
                    bias_quantizer=quantized_bits(8,0,alpha=1), 
                    activation=quantized_relu(8))(x)
        """
        x = Dense(512, activation='elu')(inputs)
        x = Dense(256, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        #
        x = Dense(256, activation='elu')(inputs)
        x = Dense(128, activation='elu')(x)
        
        x = Dense(8, activation='elu')(inputs)
        """
        z_mean = QDense(self.dim_z,
        kernel_quantizer=quantized_bits(8,0,alpha=1), 
        bias_quantizer=quantized_bits(8,0,alpha=1),
        name='z_mean', activation=None)(x)
        # instantiate encoder model
        if self.mode == 'ordered':
         self.encoder = Model([in_regression], [z_mean], name='encoder')

        self.encoder.compile(optimizer='adam', loss='mean_squared_error')
        return self.encoder


    def train_model(self, train, batch_size=10000,epochs = 5):

        earlystopper = EarlyStopping(monitor='loss', patience=50, verbose=0, min_delta=1e-7)
        checkpointer = ModelCheckpoint(self.model_filename, monitor='loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000000001)
        tensorboard = TensorBoard(log_dir='tb_logs/' + self.filename)

        lr = 1e-3
        epoch_num = 0
        best_loss = 1e10
        cur_patience = 0
        max_patience_lr = 5
        max_patience_quit = 10
        
        # Or have 2 output arms with the same output that have different inputs and train like that?
        # Or latent space that is much hgiher than inut space, as this problem is much
        # more complex?

        res_train = self.model.fit(
            train, 
            self.MSE(self.c, self.dataset_len, self.dim_z),
            #self.loss_fn(train[0].shape[0], self.dim_z),
            batch_size=batch_size, 
            epochs=epochs,
            shuffle=True,
            verbose=1 if self.verbose else 0, 
            callbacks=[earlystopper, reduce_lr, checkpointer, tensorboard]
        )

    def savemodel(self):
        if not os.path.exists('%s/savedmodels' %(self.modeldir)):
            os.makedirs('%s/savedmodels' %(self.modeldir))
        self.model.save('%s/savedmodels/%s.h5' %(self.modeldir,self.filename))

    def load_weights(self, weight_filename):
        self.model.load_weights(weight_filename)

    def evaluate_radius_max(self, train_data, batch_size):
        logger.info('evaluating radius max')

        if self.use_hls4ml:
            logger.info('using hls4ml to evaluate radius max')
            hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
            hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
            hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
            config = hls4ml.utils.config_from_keras_model(self.encoder, granularity='name')
            # config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
            # config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
            print("-----------------------------------")
            plotting.print_dict(config)
            print("-----------------------------------")
            if not os.path.exists(self.modeldir+"/hls3ml_models"):
                os.makedirs(self.modeldir+"/hls3ml_models")
            hls_model = hls4ml.converters.convert_from_keras_model(self.encoder,
                                                       hls_config=config,
                                                       output_dir=self.modeldir+"/hls3ml_models",
                                                       part='xcu250-figd2104-2L-e')
            hls_model.compile()
            self.hls4ml_model = hls_model
            latent_space_hls = hls_model.predict(np.ascontiguousarray(train_data))
            train_scores = get_R(latent_space_hls - self.MSE(self.c, latent_space_hls.shape[0], self.dim_z))
            print(train_scores)
            max_radius = np.max(train_scores)
            return max_radius




        else:
            latent_space = self.encoder.predict(train_data, batch_size=batch_size, verbose=self.verbose)
            print(latent_space)

            # logger.debug("laten space type %s" % (print(latent_space[0][0].type)))

            train_scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))
            print(train_scores)
            max_radius = np.max(train_scores)

            return max_radius


    def evaluate_radius(self, data, r_max, batch_size):
        logger.info('evaluating radius')


        if self.use_hls4ml:
            latent_space_hls = self.hls4ml_model.predict(np.ascontiguousarray(data))


            scores = get_R(latent_space_hls - self.MSE(self.c, latent_space_hls.shape[0], self.dim_z))

            scores_r = scores / r_max

            return scores_r

        else:



            latent_space = self.encoder.predict(data, batch_size=batch_size, verbose=self.verbose)

            scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))

            scores_r = scores / r_max

            return scores_r

    # def evaluate_radius_noscores(self, data, r_max, batch_size):
    #     latent_space = self.encoder.predict(data, batch_size=batch_size, verbose=self.verbose)

    #     # scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))
 
    #     scores_r = 1
    #     return scores_r

    """
    def evaluate_radius(self, test_bg, test_sig, batch_size=100):
        latent_space_bg = self.encoder.predict(test_bg, batch_size=batch_size, verbose=self.verbose)
        latent_space_sig = self.encoder.predict(test_sig, batch_size=batch_size, verbose=self.verbose)

        test_bg_scores = get_R(latent_space_bg - self.loss_fn(latent_space_bg.shape[0], self.dim_z))
        test_sig_scores = get_R(latent_space_sig - self.loss_fn(latent_space_sig.shape[0], self.dim_z))
        
        self.radius_bg = test_bg_scores
        self.radius_sig = test_sig_scores
        
        self.test_bg_scores_r = test_bg_scores / self.max_radius
        self.test_sig_scores_r = test_sig_scores / self.max_radius
        return test_bg_scores, test_sig_scores
    """


def evaluate_efficiencies(labels, events):
    fpr, tpr, _ = roc_curve(labels, events)
    
    #background efficiencies
    efficiency1 = 10.0**-2
    efficiency2 = 10.0**-3
    efficiency3 = 10.0**-4
    efficiency4 = 10.0**-5

    #epsilon values
    epsilon1 = 0.0
    epsilon2 = 0.0
    epsilon3 = 0.0
    epsilon4 = 0.0
   
    #flags to tell when done
    done1 = False
    done2 = False
    done3 = False
    done4 = False

    #iterate through bkg efficiencies and get as close as possible to the desired efficiencies.
    for i in range(len(fpr)):
        bkg_eff = fpr[i]
        if bkg_eff >= efficiency1 and done1 == False:
            epsilon1 = tpr[i]
            done1 = True
        if bkg_eff >= efficiency2 and done2 == False:
            epsilon2 = tpr[i]
            done2 = True
        if bkg_eff >= efficiency3 and done3 == False:
            epsilon3 = tpr[i]
            done3 = True
        if bkg_eff >= efficiency4 and done4 == False:
            epsilon4 = tpr[i]
            done4 = True

        if done1 and done2 and done3 and done4:
            break

    return epsilon1, epsilon2, epsilon3, epsilon4 