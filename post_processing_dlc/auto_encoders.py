# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:22:27 2023

@author: Kevin Delgado
Autoencoder classes used in post processing analysis
"""

import tensorflow as tf
import numpy as np
from skimage import transform
from post_processing_dlc import utils


class Nose2TailAutoEncoder():
    def __init__(self, model_weights_path):
        """ initialize nose2tail autoencoder tp predict missing markers from nose to tail2 """
        self.autoencoder = tf.keras.Sequential([
            tf.keras.Input(shape=(16,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
        ])
        self.autoencoder.load_weights(model_weights_path)
        
    def run(self, normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE, FRAMES):
        """ run autoencoder to predict missing body parts between nose-tail2"""
        subset_normalized_dlc_arr = normalized_dlc_arr[FRAMES, :8]
        subset_markers_above_confidence = MARKERS_ABOVE_CONFIDENCE[FRAMES,:8]
        
        subset_normalized_dlc_arr[np.invert(subset_markers_above_confidence),:] = np.array([0,0])  
        assert subset_normalized_dlc_arr.shape == normalized_dlc_arr[FRAMES, :8].shape
    
        # autoencoder inference
        ae_results = self.autoencoder.predict(subset_normalized_dlc_arr.reshape((-1, 16)))
        ae_results = ae_results.reshape((-1, 8, 2))
        
        # similarity transformation for each frame autoencoder was run on, else hungarian method
        for i in range(len(ae_results)):
            src = ae_results[i, subset_markers_above_confidence[i]]
            dest = subset_normalized_dlc_arr[i, subset_markers_above_confidence[i]]
            result, simtransform = utils.run_similarity_transformation(src, dest)
            if result:
                src_transformed = simtransform(ae_results[i, np.invert(subset_markers_above_confidence[i])])
                subset_normalized_dlc_arr[i,np.invert(subset_markers_above_confidence[i])] = src_transformed
            else:
                subset_normalized_dlc_arr[i] = utils.find_closest_neighbor(subset_normalized_dlc_arr[i], ae_results[i], subset_markers_above_confidence[i])
        
        # fit new predicted missing body parts back to normalized_dlc_arr
        normalized_dlc_arr[FRAMES, :8] = subset_normalized_dlc_arr
        MARKERS_ABOVE_CONFIDENCE[FRAMES,:8] = True
        
        return normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE


class FeetAutoEncoder():
    def __init__(self, model_weights_path):
        """ initialize feet autoencoder to predict missing feet markers """
        self.autoencoder = tf.keras.Sequential([
            tf.keras.Input(shape=(24,)),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(72, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(72, activation="relu"),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
        ])
        self.autoencoder.load_weights(model_weights_path) 
        

    def run(self, normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE, FRAMES):
        """ run autoencoder to predict missing body parts for feet"""
        subset_normalized_dlc_arr = normalized_dlc_arr[FRAMES, :]
        subset_markers_above_confidence = MARKERS_ABOVE_CONFIDENCE[FRAMES,:]
        
        subset_normalized_dlc_arr[np.invert(subset_markers_above_confidence),:] = np.array([0,0])  
        assert subset_normalized_dlc_arr.shape == normalized_dlc_arr[FRAMES, :].shape
    
        # autoencoder inference
        ae_results = self.autoencoder.predict(subset_normalized_dlc_arr.reshape((-1, 24)))
        ae_results = ae_results.reshape((-1, 12, 2))
        
        # similarity transformation for each frame autoencoder was run on, else hungarian method
        for i in range(len(ae_results)):
            src = ae_results[i, subset_markers_above_confidence[i]]
            dest = subset_normalized_dlc_arr[i, subset_markers_above_confidence[i]]
            result, simtransform = utils.run_similarity_transformation(src, dest)
            if result:
                src_transformed = simtransform(ae_results[i, np.invert(subset_markers_above_confidence[i])])
                subset_normalized_dlc_arr[i,np.invert(subset_markers_above_confidence[i])] = src_transformed
            else:
                subset_normalized_dlc_arr[i] = utils.find_closest_neighbor(subset_normalized_dlc_arr[i], ae_results[i], subset_markers_above_confidence[i])
        
        # fit new predicted missing body parts back to normalized_dlc_arr
        normalized_dlc_arr[FRAMES, :] = subset_normalized_dlc_arr
        MARKERS_ABOVE_CONFIDENCE[FRAMES,:] = True
        
        return normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE


class AllMarkerAutoEncoder():
    def __init__(self, model_weights_path):
        """ initialize all marker autoencoder to predict missing markers of mouse pose """
        self.autoencoder = tf.keras.Sequential([
            tf.keras.Input(shape=(24,)),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(72, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(72, activation="relu"),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
        ])
        self.autoencoder.load_weights(model_weights_path) 
        
        
    def run(self, normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE, FRAMES):
        """ run autoencoder to predict missing body parts for feet"""
        subset_normalized_dlc_arr = normalized_dlc_arr[FRAMES, :]
        subset_markers_above_confidence = MARKERS_ABOVE_CONFIDENCE[FRAMES,:]
        
        subset_normalized_dlc_arr[np.invert(subset_markers_above_confidence),:] = np.array([0,0])  
        assert subset_normalized_dlc_arr.shape == normalized_dlc_arr[FRAMES, :].shape
    
        # autoencoder inference
        ae_results = self.autoencoder.predict(subset_normalized_dlc_arr.reshape((-1, 24)))
        ae_results = ae_results.reshape((-1, 12, 2))
        
        # similarity transformation for each frame autoencoder was run on, else hungarian method
        for i in range(len(ae_results)):
            src = ae_results[i, subset_markers_above_confidence[i]]
            dest = subset_normalized_dlc_arr[i, subset_markers_above_confidence[i]]
            result, simtransform = utils.run_similarity_transformation(src, dest)
            if result:
                src_transformed = simtransform(ae_results[i, np.invert(subset_markers_above_confidence[i])])
                subset_normalized_dlc_arr[i,np.invert(subset_markers_above_confidence[i])] = src_transformed
            else:
                subset_normalized_dlc_arr[i] = utils.find_closest_neighbor(subset_normalized_dlc_arr[i], ae_results[i], subset_markers_above_confidence[i])
        
        # fit new predicted missing body parts back to normalized_dlc_arr
        normalized_dlc_arr[FRAMES, :] = subset_normalized_dlc_arr
        MARKERS_ABOVE_CONFIDENCE[FRAMES,:] = True
        
        return normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE