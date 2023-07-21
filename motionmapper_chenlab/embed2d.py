# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:20:47 2023

@author: Kevin Delgado
Embed data into 2D using UMAP algorithm
"""

import numpy as np
import pickle
from tqdm import tqdm

class Embed2DUMAP():
    def __init__(self, umap_model_path, scaling_parameters_path, parameters):
        """ initialize umap model """
    
        print("Loading UMAP model")
        with open(umap_model_path, 'rb') as f:
            self.um = pickle.load(f)
        self.um.negative_sample_rate = parameters['embed_negative_sample_rate']
        self.um.verbose = True
        print('Loaded UMAP Model.')
              
        self.trainparams = np.load(scaling_parameters_path, allow_pickle=True)
        print('Loaded scaling parameters.')
        
        
    def inference(self, data):
        """ embed data into 2D using model """
        
        batch_data = self.data2batches(data)
        
        num_of_batches = len(batch_data)
        batch_zval_list = []
        
        print('Finding Embeddings w/ UMAP')
        for batch_idx in tqdm(range(num_of_batches), desc='\tBatch'):
            zval = self.um.transform(batch_data[batch_idx])
            zval = zval - self.trainparams[0]
            zval = zval * self.trainparams[1]
            batch_zval_list.append(zval)
        zVals = np.concatenate(np.array(batch_zval_list), 0)
        return zVals
    
    
    def data2batches(self, data, MAX_SIZE = 10000):
        """ split up data into batches of size MAX_SIZE for tracking 
        note: 10,0000 seems the best at the moment for quick tracking of progress embedding data """
        
        if len(data) <= MAX_SIZE:
            return [data]
        num_of_batches = len(data)//MAX_SIZE
        batch_sizes = [MAX_SIZE for _ in range(num_of_batches)]
        remainder = len(data) % MAX_SIZE
        if remainder > 0:
            num_of_batches += 1
            batch_sizes.append(remainder)
            
        batch_data = np.split(data, np.cumsum(batch_sizes)[:-1])
        return batch_data
            