# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:00:38 2023

@author: Kevin Delgado

Convert projections to wavelets
"""
import sys
import numpy as np
sys.path.append(r"Z:\Dropbox\Dropbox\Chen Lab Team Folder\Projects\Home_Cage_Training\DeepLabCut\BehaviorAnalysis\motionmapperpy")
import motionmapperpy as mmpy


def convert2wavelets(projections, per_trial_length, parameters):
    """ convert data into wavelets """
    
    print("Finding Wavelets.")
    batch_projections, batch_per_trial_lengths = data2trialbatches(projections, per_trial_length)
    num_of_batches = len(batch_projections)
    
    batch_wavelet_list = []
    for batch_idx in range(len(batch_projections)):
        print("Batch {}/{}".format(str(batch_idx+1), str(num_of_batches)))
        print("Pose data shape:", batch_projections[batch_idx].shape)
        batch_wavelet, f = mmpy.mm_findWaveletsChenLab(batch_projections[batch_idx], batch_per_trial_lengths[batch_idx], parameters.pcaModes, parameters)
        batch_wavelet = batch_wavelet / np.sum(batch_wavelet, 1)[:, None]
        batch_wavelet_list.append(batch_wavelet)
    
    wavelets = np.concatenate(np.array(batch_wavelet_list), 0)
    return wavelets


def data2trialbatches(projections, per_trial_length, MAX_BATCH_SIZE = 25000):
    """ split data into individual batches based on trial lengths to not overload RAM """
    
    # split data into batches
    num_of_frames_batch = 0 

    batch_sizes = []
    batch_trial_indexes = []

    trial_indexes = []
    for i, trial_length in enumerate(per_trial_length):
        if num_of_frames_batch + trial_length > MAX_BATCH_SIZE:
            batch_trial_indexes.append(trial_indexes)
            batch_sizes.append(num_of_frames_batch)
            
            trial_indexes = [i]
            num_of_frames_batch = trial_length
        else:
            num_of_frames_batch += trial_length
            trial_indexes.append(i)
            
        if i == len(per_trial_length)-1:
            batch_sizes.append(num_of_frames_batch)
            batch_trial_indexes.append(trial_indexes)
            
    batch_projections = np.split(projections, np.cumsum(batch_sizes)[:-1])
    batch_per_trial_lengths = []
    for batch_trials in batch_trial_indexes:
        single_batch_per_trial_length = [per_trial_length[trial_idx] for trial_idx in batch_trials]
        batch_per_trial_lengths.append(single_batch_per_trial_length)
        
    return batch_projections, batch_per_trial_lengths