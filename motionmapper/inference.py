# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:34:57 2023
@author: Kevin Delgado
Container for MotionMapper process
"""

import datetime
import numpy as np
import os
import pickle
from scipy.io import loadmat, savemat
import stat
from tqdm import tqdm

from motionmapper.parameters import parameters
from motionmapper.auto_encoder import AE_Encoder
from motionmapper.wavelet_transform import wavelet_transform
from motionmapper.embed2d import Embed2DUMAP
from motionmapper.watershedregions import get_watershed_regions
from motionmapper.draw_plot import draw_plot

import ChenLabPyLib


class MotionMapperInference():
    def __init__(self, umap_model_path, auto_encoder_model_path, scaling_parameters_path, look_up_table_path, watershed_file_path, version):
        """ initialize models used in motionampper process """
        
        print("Initializing MotionMapper inference class.")
        self.umap_model_path = umap_model_path
        self.auto_encoder_model_path = auto_encoder_model_path
        self.watershed_file_path = watershed_file_path
        self.version = version

        self.encoder = AE_Encoder(model_path=self.auto_encoder_model_path)
        self.umapmodel = Embed2DUMAP(self.umap_model_path, scaling_parameters_path, parameters)
        
        # initialize watershed look-up table
        with open(look_up_table_path, 'rb') as f:
            BEHAVIOR_LABELED_LOOK_UP_TABLE = pickle.load(f)
        self.BEHAVIOR_LABELED_LOOK_UP_TABLE_INVERTED = {}
        for globalregion in BEHAVIOR_LABELED_LOOK_UP_TABLE.keys():
            for region in BEHAVIOR_LABELED_LOOK_UP_TABLE[globalregion]:
                self.BEHAVIOR_LABELED_LOOK_UP_TABLE_INVERTED[str(region)] = int(globalregion)
        print("Loaded behavior labeled look-up table")
        
        
    def run(self, pose_data, per_trial_length, mat_files_used, animalRFID, animal_folder, overwrite=False, save_progress=False, save2trialmat=False, sigma=0.9, disable_progressbar=False):
        """ run MotionMapper inference for data """
        
        print("Running MotionMapper inference.")
        mat_file_path = os.path.join(animal_folder, "ENCODED_POSE_DATA.mat")
        if os.path.exists(mat_file_path) and overwrite == False:
            encoded_pose_data = loadmat(mat_file_path)["data"]
        else:
            encoded_pose_data = self.encoder.inference(pose_data)
            if save_progress:
                savemat(mat_file_path, {"data": encoded_pose_data})
                os.chmod(mat_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                print("\tSaved encoded pose data to ENCODED_POSE_DATA.mat for {}".format(animalRFID))
        print("\tEncoded pose data dim:", encoded_pose_data.shape)
        
        mat_file_path = os.path.join(animal_folder, "WAVELETS.mat")
        if os.path.exists(mat_file_path) and overwrite == False:
            wavelets = loadmat(mat_file_path)["data"]
        else:
            wavelets = wavelet_transform(encoded_pose_data, per_trial_length, parameters)
            if save_progress:
                savemat(mat_file_path, {"data": wavelets})
                os.chmod(mat_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                print("\tSaved wavelets to WAVELETS.mat for {}".format(animalRFID)) 
        print("\tWavelet dim:", wavelets.shape)
        
        mat_file_path = os.path.join(animal_folder, "UMAP2D.mat")
        if os.path.exists(mat_file_path) and overwrite == False:
            embedded2ddata = loadmat(mat_file_path)["data"]
        else:
            embedded2ddata = self.umapmodel.inference(wavelets)
            if save_progress:
                savemat(mat_file_path, {"data": embedded2ddata})
                os.chmod(mat_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                print("\tSaved embedded 2D data to UMAP2D.mat for {}".format(animalRFID))
        print("\tEmbedded data dim:", embedded2ddata.shape)
        
        draw_plot(embedded2ddata, animalRFID, animal_folder, sigma=sigma)
    
        mat_file_path = os.path.join(animal_folder, "WATERSHEDREGIONS.mat")
        if os.path.exists(mat_file_path) and overwrite == False:
            watershedRegions = loadmat(mat_file_path)["data"]
        else:
            watershedRegions = get_watershed_regions(embedded2ddata, self.watershed_file_path, self.BEHAVIOR_LABELED_LOOK_UP_TABLE_INVERTED)
            if save_progress:
                savemat(mat_file_path, {"data": watershedRegions})
                os.chmod(mat_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                print("\tSaved watershedregions to WATERSHEDREGIONS.mat for {}".format(animalRFID)) 
        print("\tWatershedRegions dim:", watershedRegions.shape)
        
        if save2trialmat:
            # breakup data based on per_trial_length
            encoded_pose_data_batched = np.split(encoded_pose_data, np.cumsum(per_trial_length)[:-1])
            wavelets_batched = np.split(wavelets, np.cumsum(per_trial_length)[:-1])
            embedded2ddata_batched = np.split(embedded2ddata, np.cumsum(per_trial_length)[:-1])
            watershedRegions_batched = np.split(watershedRegions, np.cumsum(per_trial_length)[:-1])
            assert len(encoded_pose_data_batched) == len(wavelets_batched) == len(embedded2ddata_batched) == len(watershedRegions_batched) == len(mat_files_used)
            
            for i in tqdm(range(len(per_trial_length)), desc="\tSaving MotionMapper data", disable=disable_progressbar):
                mat_file = ChenLabPyLib.chenlab_filepaths(paths=mat_files_used[i])
                matdata = loadmat(mat_file)
                matdata["encoded_egocentricwTM"] = encoded_pose_data_batched[i]
                matdata["wavelets"] = wavelets_batched[i]
                matdata["embedded2D"] = embedded2ddata_batched[i]
                matdata["watershedregions"] = watershedRegions_batched[i]
                matdata["umap_model_path"] = self.umap_model_path
                matdata["autoencoder_model_path"] = self.auto_encoder_model_path
                matdata["motion_mapper_analyzed_ver"] = self.version
                matdata["motion_mapper_analyzed_date"] = datetime.datetime.today()
                savemat(mat_file, matdata)
                os.chmod(mat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        return 