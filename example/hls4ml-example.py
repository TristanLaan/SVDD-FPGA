#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import hls4ml
import matplotlib.pyplot as plt
import numpy as np
import plotting
import tensorflow as tf
from callbacks import all_callbacks
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Activation, BatchNormalization, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical


def main(model_dir):
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X, y = data['data'], data['target']
    logging.info("Get dataset")
    logging.debug(f"{'=' * 40}\nDATASET INFO\n{'=' * 40}\n"
                  f"{data['feature_names']=:}\n{(X.shape, y.shape)=:}\n"
                  f"{X[:5]=:}\n{y[:5]=:}")

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 5)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    logging.debug(f"{y[:5]=:}")

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    np.save('X_train_val.npy', X_train_val)
    np.save('X_test.npy', X_test)
    np.save('y_train_val.npy', y_train_val)
    np.save('y_test.npy', y_test)
    np.save('classes.npy', le.classes_)

    model = Sequential()
    model.add(Dense(64, input_shape=(16,), name='fc1',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu1'))
    model.add(Dense(32, name='fc2', kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu2'))
    model.add(Dense(32, name='fc3', kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='relu', name='relu3'))
    model.add(Dense(5, name='output', kernel_initializer='lecun_uniform',
              kernel_regularizer=l1(0.0001)))
    model.add(Activation(activation='softmax', name='softmax'))

    model_saved = model_dir / 'KERAS_check_best_model.h5'
    if model_saved.exists():
        logging.info("Loading saved model")
        model = load_model(str(model_saved))
    else:
        logging.info("Training model")
        adam = Adam(lr=0.0001)
        model.compile(optimizer=adam, loss=[
                      'categorical_crossentropy'], metrics=['accuracy'])
        callbacks = all_callbacks(stop_patience=1000,
                                  lr_factor=0.5,
                                  lr_patience=10,
                                  lr_epsilon=0.000001,
                                  lr_cooldown=2,
                                  lr_minimum=0.0000001,
                                  outputDir=str(model_dir))
        model.fit(X_train_val, y_train_val, batch_size=1024,
                  epochs=30, validation_split=0.25, shuffle=True,
                  callbacks=callbacks.callbacks)

    y_keras = model.predict(X_test)
    accuracy = accuracy_score(
        np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))
    logging.info(f"Model accuracy: {accuracy}")
    plt.figure(figsize=(9, 9))
    plotting.makeRoc(y_test, y_keras, le.classes_)
    plt.savefig('roc.png')


    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    config['Model']['ReuseFactor'] = 1
    logging.debug(f"{'=' * 40}\nHLS4ML CONFIGURATION\n"
                  f"{plotting.print_dict(config)}\n{'=' * 40}")

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir='model_1/hls4ml_prj',
        part='xcu280-fsvh2892-2L-e',
        backend='Vivado')

    hls4ml.utils.plot_model(hls_model, show_shapes=True,
                            show_precision=True, to_file=None)

    logging.info("Compiling HLS model")
    hls_model.compile()
    X_test = np.ascontiguousarray(X_test)
    y_hls = hls_model.predict(X_test)

    hls4ml_accuracy = accuracy_score(np.argmax(y_test, axis=1),
                                     np.argmax(y_hls, axis=1))
    logging.debug(f"Keras  Accuracy: {accuracy}")
    logging.debug(f"hls4ml Accuracy: {hls4ml_accuracy}")

    fig, ax = plt.subplots(figsize=(9, 9))
    plotting.makeRoc(y_test, y_keras, le.classes_)
    plt.gca().set_prop_cycle(None)  # reset the colors
    plotting.makeRoc(y_test, y_hls, le.classes_, linestyle='--')

    lines = [Line2D([0], [0], ls='-'),
             Line2D([0], [0], ls='--')]
    leg = Legend(ax, lines, labels=['keras', 'hls4ml'],
                 loc='lower right', frameon=False)
    ax.add_artist(leg)
    fig.savefig('compare.png')

    logging.info("Building HLS model")
    hls_model.build(csim=False)
    report = hls4ml.report.read_vivado_report('model_1/hls4ml_prj/')
    logging.debug(f'Vivado report: {report}')


if __name__ == '__main__':
    from time import sleep
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logging.basicConfig(level=logging.DEBUG)

    boards = hls4ml.templates.get_supported_boards_dict().keys()
    logging.info(f"Supported boards: {boards}")
    plotting.print_dict(hls4ml.templates.get_backend('VivadoAccelerator').create_initial_config())
    sleep(15)

    main(model_dir=Path('model_1'))
