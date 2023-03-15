
import numpy as np
import math
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda, Input, Dense, LeakyReLU, BatchNormalization, Concatenate, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from scipy.stats import multivariate_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1

import yaml

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
import os
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
    def __init__(self, hidden_layers, filename, D, dataset_len, dim_z, c, mode=None, verbose=False,modeldir="models",quantised=False,hls4ml=False,modelpath=False,ap_fixed_width=32,ap_fixed_int=6,Flags = None):
        self.D = D
        self.dataset_len = dataset_len
        self.dim_z = dim_z
        self.c = c
        self.verbose = verbose
        self.hls4ml_model = None
        self.model = None
        self.ap_fixed_width = ap_fixed_width
        self.ap_fixed_int = ap_fixed_int
        if Flags:
            self.Flags = Flags

        self.model_filename = filename + '.h5'
        self.modeldir = str(modeldir)
        self.modelpath = modelpath
        self.use_hls4ml = hls4ml
        self.filename = filename
        self.mode = mode
        self.hidden_layers = hidden_layers


        logger.info("using hls4ml wrapper: %s" % (self.use_hls4ml))

        # if hls4ml and quantised:

        if self.modelpath:
            if "models_conventional" in self.modeldir:
                logger.info("Building NORMAL model")
                model = self.build_lhcdata_model()
                model.load_weights(self.modelpath)
            elif quantised:
                logger.debug('Loading quantised model: %s' %(self.modelpath))
                model = self.load_lhcdata_model_quantized()
            else:
                logger.debug('Loading standard model: %s' %(self.modelpath))
                model = self.load_lhcdata_model()
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
        print(self.modelpath)
        model = load_model(self.modelpath, custom_objects=co)
        model.summary()

        self.model = model
        return model

    def load_lhcdata_model(self):
        model = load_model(self.modelpath)
        logger.debug('loaded model with summary: %s' %(self.modelpath))
        model.summary()

        self.model = model
        return model

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
            self.model = Model([in_regression], [z_mean], name='encoder')

        # self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model


    def build_lhcdata_model_quantised(self):
        model = Sequential()
        
        if self.mode == 'ordered':
            inputs = Input(shape=(self.D,), name='in_regression')


        model.add(QDense(self.hidden_layers[0], input_shape=(self.D,), name='fc1',
                 kernel_quantizer=quantized_bits(self.bits,0,alpha=1), 
                 bias_quantizer=quantized_bits(self.bits,0,alpha=1),
                 kernel_initializer='lecun_uniform', 
                 kernel_regularizer=l1(0.0001))
                 )

        model.add(QActivation(activation=quantized_relu(self.bits), name='relu1'))
        if len(self.hidden_layers) > 1:
            for i, v in enumerate(self.hidden_layers):
                if (i > 0):

                    model.add(QDense(v, name='fc%s' %(str(i+1)),
                        kernel_quantizer=quantized_bits(self.bits,0,alpha=1), 
                        bias_quantizer=quantized_bits(self.bits,0,alpha=1),
                        kernel_initializer='lecun_uniform', 
                        kernel_regularizer=l1(0.0001))
                    )
                    model.add(QActivation(activation=quantized_relu(self.bits),  name='relu%s' %(str(i+1))))

        """
        x = Dense(512, activation='elu')(inputs)
        x = Dense(256, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        #
        x = Dense(256, activation='elu')(inputs)
        x = Dense(128, activation='elu')(x)
        
        x = Dense(8, activation='elu')(inputs)
        """
        i = len(self.hidden_layers) + 1
        model.add(QDense(self.dim_z, name='fc%s' %(str(i+2)),
                        kernel_quantizer=quantized_bits(self.bits,0,alpha=1), 
                        bias_quantizer=quantized_bits(self.bits,0,alpha=1),
                        kernel_initializer='lecun_uniform', 
                        kernel_regularizer=l1(0.0001))
                    )

        # model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model



    def savemodel(self,filepath,model=None):
        logger.info("saving model %s" % (filepath))
        if os.path.exists(filepath):
            os.remove(filepath)
        if model:
            model.save(filepath)
        else:
            self.model.save(filepath)

    def train_model(self, train, batch_size=10000,epochs = 5):
        outdir = '%s/savedmodels' %(self.modeldir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = os.path.join(outdir,self.model_filename)
        print(outpath)

        
        earlystopper = EarlyStopping(monitor='loss', patience=50, verbose=0, min_delta=1e-7)
        # checkpointer = ModelCheckpoint(filepath=outpath, monitor='loss', verbose=1,save_weights_only=False, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000000001)
        # tensorboard = TensorBoard(log_dir='tb_logs/' + self.filename)

        callbacks=[earlystopper, reduce_lr]
        #SparseModel
        if self.Flags.sparsity:
            from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
            from tensorflow_model_optimization.sparsity.keras import strip_pruning
            pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(float(self.Flags.sparsity), begin_step=1, frequency=100)}
            model = prune.prune_low_magnitude(self.model, **pruning_params)
            self.model = model
            callbacks.append(pruning_callbacks.UpdatePruningStep())





        lr = 1e-3
        epoch_num = 0
        best_loss = 1e10
        cur_patience = 0
        max_patience_lr = 5
        max_patience_quit = 10
        
        # Or have 2 output arms with the same output that have different inputs and train like that?
        # Or latent space that is much hgiher than inut space, as this problem is much
        # more complex?
        self.model.compile(optimizer='adam', loss='mean_squared_error')


        
        



        print(self.model.get_weights()[0])
        res_train = self.model.fit(
            train, 
            self.MSE(self.c, self.dataset_len, self.dim_z),
            #self.loss_fn(train[0].shape[0], self.dim_z),
            batch_size=batch_size, 
            epochs=epochs,
            shuffle=True,
            verbose=1 if self.verbose else 0, 
            callbacks=callbacks
        )

        if self.Flags.sparsity:
            # Save the model again but with the pruning 'stripped' to use the regular layer types
            model = strip_pruning(self.model)
            self.model = model
            self.savemodel(outpath)
        else:
            self.model = model

        
        for l,layer in enumerate(self.model.layers):
            print(layer.name, layer)
            w = self.model.get_weights()[l]
            h, b = np.histogram(w, bins=100)
            plt.figure(figsize=(7,7))
            plt.bar(b[:-1], h, width=b[1]-b[0])
            plt.semilogy()
            print('% of zeros = {}'.format(np.sum(w==0)/np.size(w)))
            plt.savefig(outpath.replace(".h5","layer_%s_%s.pdf" %(l,layer.name)))
            plt.clf()

        self.savemodel(outpath)




    def load_weights(self, weight_filename):
        self.model.load_weights(weight_filename)

    def evaluate_radius_max(self, train_data, batch_size):
        logger.info('Entering radius max function')


        if self.use_hls4ml:
            logger.info('using hls4ml to evaluate radius max')

            # hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
            # hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
            # hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
            config = hls4ml.utils.config_from_keras_model(self.model,granularity='name')
            # config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
            # config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
            ap_fixed = "<" + str(self.ap_fixed_width)+","+ str(self.ap_fixed_int) + ">"
            config['Model']['Precision'] = 'ap_fixed' + ap_fixed
            # plotting.print_dict(config)
            config.pop('LayerName')


            # config['LayerName']['in_regression']['Precision']['result'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['dense']['Precision']['weight'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense']['Precision']['result'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense']['Precision']['bias'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['dense_elu']['Precision'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_elu']['table_t']= 'ap_fixed' + ap_fixed

            # config['LayerName']['dense_1']['Precision']['weight'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_1']['Precision']['result'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_1']['Precision']['bias'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['dense_1_elu']['Precision'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_1_elu']['table_t'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['dense_2']['Precision']['weight'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_2']['Precision']['result'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_2']['Precision']['bias'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['dense_2_elu']['Precision'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['dense_2_elu']['table_t'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['z_mean']['Precision']['weight'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['z_mean']['Precision']['result'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['z_mean']['Precision']['bias'] = 'ap_fixed' + ap_fixed

            # config['LayerName']['z_mean_linear']['Precision'] = 'ap_fixed' + ap_fixed
            # config['LayerName']['z_mean_linear']['table_t']= 'ap_fixed' + ap_fixed


            plotting.print_dict(config)
            
            print("-----------------------------------")
            self.hls4ml_model_folder = self.modeldir+"/hls4ml_models" + "_" + str(self.ap_fixed_width)+"_"+ str(self.ap_fixed_int) +  "/"+self.filename
            if not os.path.exists(self.modeldir+"/hls4ml_models" + "_" + str(self.ap_fixed_width)+"_"+ str(self.ap_fixed_int)):
                os.makedirs(self.modeldir+"/hls4ml_models" + "_" + str(self.ap_fixed_width)+"_"+ str(self.ap_fixed_int))
            with open(self.modeldir+"/hls4ml_models" + "_" + str(self.ap_fixed_width)+"_"+ str(self.ap_fixed_int) +"/hlsmodel.yml", 'w') as file:
                documents = yaml.dump(config, file)

            print(self.hls4ml_model_folder)
            if os.path.exists(self.hls4ml_model_folder):
                os.system("rm -r %s" % (self.hls4ml_model_folder))
                os.makedirs(self.hls4ml_model_folder)
            else:
                os.makedirs(self.hls4ml_model_folder)

            hls_model = hls4ml.converters.convert_from_keras_model(self.model,
                                                       hls_config=config,
                                                       output_dir=self.hls4ml_model_folder,
                                                       part='xcu250-figd2104-2L-e')

            hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=self.hls4ml_model_folder+"/plot.pdf")
            logger.info('compiling the hls_model')
            hls_model.compile()
            self.hls4ml_model = hls_model



            if self.Flags.build:
                logger.info('building the hls_model')
                hls_model.build(csim=False)
                return

            logger.info('predicting..')
            latent_space_hls = hls_model.predict(train_data)
            latent_space = latent_space_hls
            logger.info('getting training scores..')
            train_scores = get_R(latent_space_hls - self.MSE(self.c, latent_space_hls.shape[0], self.dim_z))
            max_radius = np.max(train_scores)

        else:
            logger.info('using keras to evaluate radius max')

            latent_space = self.model.predict(train_data, batch_size=batch_size, verbose=self.verbose)

            train_scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))
            max_radius = np.max(train_scores)

        logger.info('previewing training latent space..')
        print(latent_space)
        return max_radius


    def evaluate_radius(self, data, r_max, batch_size):
        logger.info('evaluating radius')


        if self.use_hls4ml:
            logger.info('using hls4ml: to evaluate radius')

            latent_space_hls = self.hls4ml_model.predict(np.ascontiguousarray(data))

            logger.info('using hls4ml: getting scores')

            scores = get_R(latent_space_hls - self.MSE(self.c, latent_space_hls.shape[0], self.dim_z))

            scores_r = scores / r_max

            return scores_r

        else:

            logger.info('using keras to evaluate radius')

            latent_space = self.model.predict(data, batch_size=batch_size, verbose=self.verbose)

            scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))

            scores_r = scores / r_max

            return scores_r

    # def evaluate_radius_noscores(self, data, r_max, batch_size):
    #     latent_space = self.model.predict(data, batch_size=batch_size, verbose=self.verbose)

    #     # scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))
 
    #     scores_r = 1
    #     return scores_r

    """
    def evaluate_radius(self, test_bg, test_sig, batch_size=100):
        latent_space_bg = self.model.predict(test_bg, batch_size=batch_size, verbose=self.verbose)
        latent_space_sig = self.model.predict(test_sig, batch_size=batch_size, verbose=self.verbose)

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