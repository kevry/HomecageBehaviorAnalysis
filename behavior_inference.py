# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:01:17 2023

@author: Kevin Delgado
"""

import argparse
import time
import numpy as np
import os

def get_args():
    """ gets arguments from command line """
    parser = argparse.ArgumentParser(
        description="Parsing argument for animal ID",
        epilog="python file.py --rfid WD2123BV"
    )
    # required argument
    parser.add_argument("--rfid", '-i', required=False, help='RFID of animal to run analysis on')
    args = parser.parse_args()
    return args.rfid


if __name__ == "__main__":
    
    from motionmapper_chenlab.motionmapper_infer import MotionMapperInference
    from extract_trials_datajoint import extract_trials_datajoint
    from post_processing_dlc.post_processing_dlc import PostAnalysisDLC
    import config as cfg
    
    auto_encoder_model_path = cfg .motion_mapper_file_paths["auto_encoder_model_path"]
    scaling_parameters_path = cfg.motion_mapper_file_paths["scaling_parameters_path"]
    umap_model_path = cfg.motion_mapper_file_paths["umap_model_path"]
    look_up_table_path = cfg.motion_mapper_file_paths["look_up_table_path"]
    watershed_file_path = cfg.motion_mapper_file_paths["watershed_file_path"]
    processing_folder = cfg.processing_folder
    
    # animal RFID
    animalRFID = "000C95218038"
    #animalRFID = get_args()
    
    # create animal folder
    animal_folder = os.path.join(processing_folder, animalRFID)
    os.makedirs(animal_folder, exist_ok=True)
    
    # create instance of MotionMapperInference object
    mminfer = MotionMapperInference(
        umap_model_path=umap_model_path, 
        auto_encoder_model_path=auto_encoder_model_path, 
        scaling_parameters_path=scaling_parameters_path,
        look_up_table_path=look_up_table_path,
        watershed_file_path=watershed_file_path
    )
    
    # create instance of Post-Process DLC object
    postdlc = PostAnalysisDLC()
    
    start_time = time.time()
    
    # %% Get all trials from DataJoint. Save all trials with matching .mat file into found_trials.csv and missing trials into missing_trials.csv
    
    found_trials_csv_path = extract_trials_datajoint(
        animalRFID=animalRFID, 
        animal_folder=animal_folder, 
        save_missing_trials=True, 
        overwrite=True
    )
    
    # %% Post-Processing DeepLabCut data for downstream behavior analysis
    
    post_analyzed_dlc_file_path = postdlc.run(
        csv_path=found_trials_csv_path, 
        animalRFID=animalRFID, 
        animal_folder=animal_folder,
        overwrite=True,
        save2trialmat=False
    )

    # %% Load in post-processed DeepLabCut data
    
    raw_data = np.load(post_analyzed_dlc_file_path)
    projections = raw_data['data']
    print("Pose data dim:", projections.shape)
    
    projections_flatten = projections.reshape((-1, 36))
    per_trial_length = raw_data['per_trial_length']
    del raw_data, projections
    print("Pose data reshaped:", projections_flatten.shape)
    
    # %% Run MotionMapper process
    
    mminfer.run(
        pose_data=projections_flatten, 
        per_trial_length=per_trial_length,
        mat_files_used=None,
        animalRFID=animalRFID, 
        animal_folder=animal_folder,
        sigma=0.5,
        save_progress=True,
        save2trialmat=False
    )
    
    # %%
    print("Elapsed time:", time.time() - start_time)
