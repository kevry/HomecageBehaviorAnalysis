# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:23:01 2023

@author: Kevin Delgado

Post processing for DeepLabCut data
"""

import os
import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal
import stat

from post_processing_dlc.auto_encoders import Nose2TailAutoEncoder, FeetAutoEncoder, AllMarkerAutoEncoder
from post_processing_dlc import utils


class PostAnalysisDLC():
    def __init__(self, nose2tail_ae_path, feet_ae_path, all_ae_path, CONFIDENCE_THRESH=0.7):
        """ initialize post processing class """
        
        self.nose2tail_ae_path = nose2tail_ae_path
        self.feet_ae_path = feet_ae_path
        self.all_ae_path = all_ae_path
        self.CONFIDENCE_THRESH = CONFIDENCE_THRESH
        self.initialize_autoencoders()
        
        
    def initialize_autoencoders(self):
        """ initialize autoencoders used in post processing """
        
        self.nose2tailAE = Nose2TailAutoEncoder(self.nose2tail_ae_path)
        self.feetAE = FeetAutoEncoder(self.feet_ae_path)
        self.allmarkerAE = AllMarkerAutoEncoder(self.all_ae_path)


    def run(self, csv_path, animalRFID, animal_folder, overwrite=False, save2trialmat=False):
        """ run post-processing on DeepLabCut data """
        print("\nPost-Processing DeepLabCut data.")
        
        npz_file_path = os.path.join(animal_folder, "POST_ANALYZED_DLC.npz")
        if overwrite == True:
            if os.path.exists(npz_file_path):
                print("\tOverwritting POST_ANALYZED_DLC.npz for {}".format(animalRFID))
        else:
            if os.path.exists(npz_file_path):
                print("\tFound POST_ANALYZED_DLC.npz for {}".format(animalRFID))
                return npz_file_path

        # get list of mat files
        trial_data = pd.read_csv(csv_path).values
        mat_file_list = [trial[1] for trial in trial_data]
        print("\tNumber of mat files in FOUND_TRIALS.csv:", len(mat_file_list))

        # read all individual mat files
        RAW_SUBJECT_DATA = []
        RAW_SUBJECT_CONFIDENCE = []
        SUBJECT_DATA = []
        NUM_OF_FRAMES_PER_TRIAL = []
        MAT_FILES_USED = []
        PROCESSED_START_END_INDEXES_PER_TRIAL = []
        
        ERROR_TRIALS = []
        ERROR_TRIALS2 = []
        MAT_FILES_NOT_USED = []

        try: # skip entire animal dataset if error occurs
            for mat_file in tqdm(mat_file_list, desc="\tPost-processing data"):  
                OUTPUT = self.pose_post_processing_per_video_TM(mat_file)
        
                if OUTPUT["valid_clip"]:
                    RAW_SUBJECT_DATA.append(OUTPUT["original_dlc_arr"])
                    RAW_SUBJECT_CONFIDENCE.append(OUTPUT["confidence_arr"])
                    SUBJECT_DATA.append(OUTPUT["processed_dlc_arr"])
                    NUM_OF_FRAMES_PER_TRIAL.append(OUTPUT["num_of_frames"])
                    MAT_FILES_USED.append(OUTPUT["mat_file"])
                    PROCESSED_START_END_INDEXES_PER_TRIAL.append(OUTPUT["processed_start_end_index"])
                else:
                    ERROR_TRIALS.append(OUTPUT["MARKERS_ABOVE_CONFIDENCE"])
                    ERROR_TRIALS2.append(OUTPUT["confidence_arr"])
                    MAT_FILES_NOT_USED.append(mat_file)
                    
            print("\n\t{}/{} clips used".format(len(MAT_FILES_USED), len(mat_file_list)))
        except:
            print("\tError with {}. Skipping to next set".format(animalRFID))
            traceback.print_exc()
            return None

        # # create error trial information
        # mat_file_path = os.path.join(animal_folder, "ERROR_TRIALS_POST.mat")
        # savemat(mat_file_path, {"errormark": ERROR_TRIALS})
        
        # # create error trial information
        # mat_file_path = os.path.join(animal_folder, "ERROR_TRIALS2_POST.mat")
        # savemat(mat_file_path, {"errorconf": ERROR_TRIALS2})

        if save2trialmat:
            # save processed data into mat files used
            for i, mat_file in enumerate(tqdm(MAT_FILES_USED, desc="\tSaving data to mat files")):
                # load in mat file data
                mat_file = utils.linux2windowspath(mat_file)
                matdata = loadmat(mat_file)
                
                # append egocentric data and start/end indices into mat file
                matdata["post_processed_success"] = True
                matdata["egocentricwTM"] = SUBJECT_DATA[i]
                matdata["egocentric_start_end_index"] = PROCESSED_START_END_INDEXES_PER_TRIAL[i]
                savemat(mat_file, matdata)
                os.chmod(mat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
            for i, mat_file in enumerate(tqdm(MAT_FILES_NOT_USED, desc="\tMarking mat files not used")):
                # load in mat file data
                mat_file = utils.linux2windowspath(mat_file)
                matdata = loadmat(mat_file)
                matdata["post_processed_success"] = False
                savemat(mat_file, matdata)
                os.chmod(mat_file, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            
        # concatenate data into long array
        RAW_SUBJECT_DATA = np.concatenate(RAW_SUBJECT_DATA, axis=0)
        RAW_SUBJECT_CONFIDENCE = np.concatenate(RAW_SUBJECT_CONFIDENCE, axis=0)
        SUBJECT_DATA = np.concatenate(SUBJECT_DATA, axis=0)
    
        # save post-analyzed data
        npz_file_path = os.path.join(animal_folder, "POST_ANALYZED_DLC.npz")
        np.savez(npz_file_path, data=SUBJECT_DATA, per_trial_length=NUM_OF_FRAMES_PER_TRIAL, 
                  mat_files=MAT_FILES_USED, raw=RAW_SUBJECT_DATA, conf=RAW_SUBJECT_CONFIDENCE)
        os.chmod(npz_file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        print("\tCreated POST_ANALYZED_DLC.npz file for {}!".format(animalRFID))
        return npz_file_path
        

    def pose_post_processing_per_video_TM(self, mat_file):
        """ 
            Main script for post-analyzing DeepLabCut data before running behavior analysis(MotionMapper)
            return data structure
        
            return = {
                "valid_clip": Whether clip is valid for further processing
                "mat_file": Name of file used
                "num_of_frames": Number of frames used in clip
                "original_dlc_arr": original DLC data from mat file
                "confidence_arr": array of confidence levels for each marker in each frame
                "processd_dlc_arr": processed DLC data
            }
        """
        
        # load in mat file data
        mat_file = utils.linux2windowspath(mat_file)
        data = loadmat(mat_file)
        dlc_arr = data['dlcdata']
        tm_position = data['tm_markers']
        
        # separate confidence array with marker coordinates
        dlc_confidence_arr = dlc_arr[:, :, 2].copy()
        dlc_arr = np.delete(dlc_arr, 2, axis=2)
        
        # get frame information
        original_frame_dims, scale2standard, obj_position, add2y = utils.extract_frame_info(tm_position = tm_position)
        
        # remove unnecessary markers for downstream
        tm_position = np.delete(tm_position, np.array([3,4,5]), axis = 0)
        
        # create output ditionary
        OUTPUT = {
            "valid_clip": True,
            "mat_file": os.path.basename(mat_file),
            "num_of_frames": 0,
            "original_dlc_arr": dlc_arr,
            "confidence_arr": dlc_confidence_arr,
            "processed_dlc_arr": None
        }
        
        # normalize data between 0-1 using original_frame_dims
        normalized_dlc_arr = dlc_arr.copy()
        normalized_dlc_arr[:,:,0] = normalized_dlc_arr[:,:,0]/original_frame_dims[0]
        normalized_dlc_arr[:,:,1] = normalized_dlc_arr[:,:,1]/original_frame_dims[1]
        
        # boolean matrix with markers above confidence threshold
        MARKERS_ABOVE_CONFIDENCE = dlc_confidence_arr >= self.CONFIDENCE_THRESH
        
        # search for frames that are "valid" meaning there are >= 4 markers with high confidence
        VALID_FRAME_ARR = MARKERS_ABOVE_CONFIDENCE.sum(axis=1) >= 4
        
        # skip if less than 5 valid frames in clip
        if VALID_FRAME_ARR.sum() < 5: # set dlc_arr to zeros
            OUTPUT["valid_clip"] = False
            OUTPUT["MARKERS_ABOVE_CONFIDENCE"] = MARKERS_ABOVE_CONFIDENCE
            return OUTPUT
        
        # # front cutoff
        # FRONT_CUTOFF_IDX = utils.search_front_cutoff(bool_arr=VALID_FRAME_ARR.copy())
        
        # # get cutoff of trial
        # END_CUTOFF_IDX = utils.search_cutoff(bool_arr = VALID_FRAME_ARR.copy())
        
        # # trim data past cutoff point
        # MARKERS_ABOVE_CONFIDENCE = MARKERS_ABOVE_CONFIDENCE[FRONT_CUTOFF_IDX:END_CUTOFF_IDX]

        # normalized_dlc_arr = normalized_dlc_arr[FRONT_CUTOFF_IDX:END_CUTOFF_IDX]
        # VALID_FRAME_ARR = VALID_FRAME_ARR[FRONT_CUTOFF_IDX:END_CUTOFF_IDX]
        
        # first scan through trial using linear interpolation
        interpolated_dlc_arr, MARKERS_ABOVE_CONFIDENCE = utils.interpolation(normalized_dlc_arr.copy(), MARKERS_ABOVE_CONFIDENCE.copy(), 
                                                                       threshold_for_interp = 3)
        normalized_dlc_arr = interpolated_dlc_arr.copy()

        # search for frames which there is more than 3 markers and less than 8 with high confidence between nose-tail2 and no feets predicted
        NOSE2TAIL_MARKERS = (MARKERS_ABOVE_CONFIDENCE[:, :8].sum(axis = 1) > 3) & (MARKERS_ABOVE_CONFIDENCE[:, :8].sum(axis = 1) < 8) & (MARKERS_ABOVE_CONFIDENCE[:, 8:].sum(axis = 1) == 0)

        # set markers below confidence threshold to [0,0] 
        if np.any(NOSE2TAIL_MARKERS): # autoencoder for nose-tail2 denoising
            normalized_dlc_arr_NT, MARKERS_ABOVE_CONFIDENCE_NT = self.nose2tailAE.run(normalized_dlc_arr.copy(), MARKERS_ABOVE_CONFIDENCE, NOSE2TAIL_MARKERS.copy())
            assert MARKERS_ABOVE_CONFIDENCE_NT.shape == MARKERS_ABOVE_CONFIDENCE.shape
            assert normalized_dlc_arr_NT.shape == normalized_dlc_arr.shape
            
            normalized_dlc_arr = normalized_dlc_arr_NT.copy()
            MARKERS_ABOVE_CONFIDENCE = MARKERS_ABOVE_CONFIDENCE_NT.copy()
            del normalized_dlc_arr_NT
            del MARKERS_ABOVE_CONFIDENCE_NT
            
        # median filter to remove unnecessary outliers
        for i in range(normalized_dlc_arr.shape[1]):
            normalized_dlc_arr[:,i,0] = signal.medfilt(normalized_dlc_arr[:,i,0], kernel_size=3)
            normalized_dlc_arr[:,i,1] = signal.medfilt(normalized_dlc_arr[:,i,1], kernel_size=3)

        # search for frames which nose-tail2 markers are all found
        NOSE2TAIL_MARKERS = (MARKERS_ABOVE_CONFIDENCE[:, :8].sum(axis = 1) == 8) & (MARKERS_ABOVE_CONFIDENCE[:, 8:].sum(axis = 1) == 0)
        if np.any(NOSE2TAIL_MARKERS): 
            # autoencoder for feet
            normalized_dlc_arr_FT, MARKERS_ABOVE_CONFIDENCE_FT = self.feetAE.run(normalized_dlc_arr.copy(), MARKERS_ABOVE_CONFIDENCE, NOSE2TAIL_MARKERS.copy())
            assert MARKERS_ABOVE_CONFIDENCE_FT.shape == MARKERS_ABOVE_CONFIDENCE.shape
            assert normalized_dlc_arr_FT.shape == normalized_dlc_arr.shape
            
            normalized_dlc_arr = normalized_dlc_arr_FT.copy()
            MARKERS_ABOVE_CONFIDENCE = MARKERS_ABOVE_CONFIDENCE_FT.copy()
            del normalized_dlc_arr_FT
            del MARKERS_ABOVE_CONFIDENCE_FT  
            
        # now look for frames where there is are feet markers
        NOSE2TAIL_AND_FEET_MARKERS = (MARKERS_ABOVE_CONFIDENCE[:, :8].sum(axis = 1) >= 3) & (MARKERS_ABOVE_CONFIDENCE[:, 8:].sum(axis = 1) >= 1)
        # set markers below confidence threshold to [0,0] 
        if np.any(NOSE2TAIL_AND_FEET_MARKERS): # autoencoder for nose-tail2 denoising
            # autoencoder fo frames with feet high confidence
            normalized_dlc_arr_NT, MARKERS_ABOVE_CONFIDENCE_NT = self.allmarkerAE.run(normalized_dlc_arr.copy(), MARKERS_ABOVE_CONFIDENCE, NOSE2TAIL_AND_FEET_MARKERS.copy())
            assert MARKERS_ABOVE_CONFIDENCE_NT.shape == MARKERS_ABOVE_CONFIDENCE.shape
            assert normalized_dlc_arr_NT.shape == normalized_dlc_arr.shape
            
            normalized_dlc_arr = normalized_dlc_arr_NT.copy()
            MARKERS_ABOVE_CONFIDENCE = MARKERS_ABOVE_CONFIDENCE_NT.copy()
            del normalized_dlc_arr_NT
            del MARKERS_ABOVE_CONFIDENCE_NT
            
        # do last scan of linear interpolation
        normalized_dlc_arr, MARKERS_ABOVE_CONFIDENCE = utils.interpolation(normalized_dlc_arr.copy(), MARKERS_ABOVE_CONFIDENCE.copy(), 
                                                                       threshold_for_interp = 5)
        interpolated_dlc_arr, MARKERS_ABOVE_CONFIDENCE = utils.interpolation(normalized_dlc_arr.copy(), MARKERS_ABOVE_CONFIDENCE.copy(), 
                                                                       threshold_for_interp = 5)
        normalized_dlc_arr = interpolated_dlc_arr.copy()
        
        # align data using similarity transformation with training module
        dlc_arr_og_space = normalized_dlc_arr.copy()
        dlc_arr_og_space[:, :, 0]  = (dlc_arr_og_space[:, :, 0] * original_frame_dims[0]) + (obj_position[0]*640)
        dlc_arr_og_space[:, :, 1]  = (dlc_arr_og_space[:, :, 1] * original_frame_dims[1]) + (obj_position[1]*360) - add2y
        tm_position_og_space = tm_position.copy()
        dlc_arr_tm_aligned = utils.alignbyTM(dlc_arr_og_space.copy(), tm_position_og_space.copy(), ATTACH_TM = True)   

        # move data to egocentric coordinates
        egocentric_dlc_arr = utils.egocenterALL(dlc_arr_tm_aligned.copy())
        
        # find if batch of valid frames exist
        create_batch, batch_indexes = utils.batch_for_trial(MARKERS_ABOVE_CONFIDENCE.copy(), MINIMUM_BATCH_SIZE=5)

        if create_batch:
            egocentric_dlc_arr = egocentric_dlc_arr[batch_indexes[0]:batch_indexes[-1]+1]
            MARKERS_ABOVE_CONFIDENCE = MARKERS_ABOVE_CONFIDENCE[batch_indexes[0]:batch_indexes[-1]+1]

        if np.any(np.isnan(egocentric_dlc_arr)) == True:
            raise ValueError("NaN present in processed DLC error for {}".format(os.path.basename(mat_file)))
        
        # all markers after post-processing should be switched to true, else error
        if np.all(MARKERS_ABOVE_CONFIDENCE) == False:
            OUTPUT["valid_clip"] = False 
            OUTPUT["MARKERS_ABOVE_CONFIDENCE"] = MARKERS_ABOVE_CONFIDENCE
        else: 
            # situations where markers stretch outside of training module. we can fix by simply subtracting from main
            egocentric_dlc_arr = utils.fix_stretch(egocentric_dlc_arr.copy())
            OUTPUT["processed_dlc_arr"] = egocentric_dlc_arr
            OUTPUT["num_of_frames"] = egocentric_dlc_arr.shape[0]
            OUTPUT["valid_clip"] = True
            OUTPUT["MARKERS_ABOVE_CONFIDENCE"] = MARKERS_ABOVE_CONFIDENCE
            OUTPUT["processed_start_end_index"] = batch_indexes
            
        return OUTPUT