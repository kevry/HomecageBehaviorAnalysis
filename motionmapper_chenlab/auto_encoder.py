# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:18:01 2023

@author: Kevin Delgado

AutoEncoder model for post-analysis
"""

import tensorflow as tf

class AE_Encoder():
    def __init__(self, model_path):
        """ initialize autoencoder model used for MotionMapper process """
        
        # self.AE = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=(36,)),
        #     tf.keras.layers.Dense(27, activation='relu'),
        #     tf.keras.layers.Dense(18, activation=None),
        #     tf.keras.layers.Dense(27, activation='relu'),
        #     tf.keras.layers.Dense(36, activation=None)
        # ])
        
        # load in only encoder portion of model
        AE = tf.keras.models.load_model(model_path)
        self.AEencoder = tf.keras.models.Model(inputs=AE.input, outputs=AE.layers[1].output)
        
    def inference(self, data):
        """ encode data """
        print('\tEncoding pose data.')
        encoded_data = self.AEencoder.predict(data, verbose=1)
        return encoded_data

        