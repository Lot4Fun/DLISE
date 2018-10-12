#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os

from ..lib import optimizer

import keras
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers.core import Dropout
from keras import optimizers
import tensorflow as tf

from logging import DEBUG, INFO
from logging import getLogger

# Set logger
logger = getLogger('impulso')

logger.info(tf.__version__)
logger.info(keras.__version__)

# Set HOME directory.
IMPULSO_HOME = os.environ['IMPULSO_HOME']


class ImpulsoNet(object):

    def __init__(self, exec_type, hparams):
        logger.info('Begin init of ImpulsoNet')
        self.exec_type = exec_type
        self.hparams = hparams


    def create_model(self):

        logger.info('Begin to create ImpulsoNet model')

        logger.info('Input layer')
        inputs = Input(shape=(6,))
        
        logger.info('Full Connections')
        x = Dense(7, activation='relu', name='dense1')(inputs)
        x = Dense(7, activation='relu', name='dense2')(x)
        x = Dense(7, activation='relu', name='dense3')(x)
        x = Dense(7, activation='relu', name='dense4')(x)
        x = Dense(7, activation='relu', name='dense5')(x)
        x = Dense(7, activation='relu', name='dense6')(x)
        x = Dense(7, activation='relu', name='dense7')(x)
        x = Dense(7, activation='relu', name='dense8')(x)
        x = Dense(7, activation='relu', name='dense9')(x)
        x = Dense(7, activation='relu', name='dense10')(x)
        x = Dense(7, activation='relu', name='dense11')(x)
        x = Dense(7, activation='relu', name='dense12')(x)
        x = Dense(7, activation='relu', name='dense13')(x)
        x = Dense(7, activation='relu', name='dense14')(x)
        x = Dense(7, activation='relu', name='dense15')(x)
        x = Dense(7, activation='relu', name='dense16')(x)
        x = Dense(7, activation='relu', name='dense17')(x)
        x = Dense(7, activation='relu', name='dense18')(x)
        x = Dense(7, activation='relu', name='dense19')(x)
        x = Dense(7, activation='relu', name='dense20')(x)

        logger.info('Output layer')
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)

        logger.info('Create model')
        self.model = Model(inputs=inputs, outputs=predictions)

        logger.info('Finish creating ImpulsoNet model')


    def select_optimizer(self):
        logger.info('Select optimizer')
        self.selected_optimizer = optimizer.select_optimizer(self.hparams[self.exec_type]['optimizer'])
    

    def compile(self):
        logger.info('Compile model')
        self.model.compile(optimizer=self.selected_optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        self.model.summary()
    

if __name__ == '__main__':
    pass
    