# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:25:53 2023
@author: Kevin Delgado
Use this helper to collect animal data and split into job array
"""

import argparse
import json
import os
import subprocess
import sys
import yaml

import chenlabpylib

def get_args():
    """ gets arguments from command line """
    parser = argparse.ArgumentParser(
        description="Getting configuration file for further analysis",
        epilog="python file.py --config_file_name config_test.yaml"
    )
    # required argument
    parser.add_argument("--config_file_name", '-cfg', required=True, help='File name of configuration file')
    args = parser.parse_args()
    return args.config_file_name


if __name__ == "__main__":
    
    # number of animals to run analysis per job
    batch_size = 1
    
    # get configuration file path
    # note: assuming config file is already in "configs" folder
    config_file_name = get_args()
    config_file_path = os.path.join(os.path.dirname(os.getcwd()), "configs", config_file_name)
    
    with open(config_file_path, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.exit(exc)
            
    # get animal list from cfg file
    animal_list = [str(animalRFID)for animalRFID in cfg["animal_list"]] 
    
    # get processing folder
    processing_folder = chenlabpylib.chenlab_filepaths(path=cfg["processing_folder"])
    
    if len(animal_list) <= 0:
        raise ValueError("No animals listed in configuration file.")
    
    if animal_list[0] == "all":
        print("Using all animals in {}".format(processing_folder))
        animal_list = os.listdir(processing_folder)
    
    # separate list of videos in chunks
    animal_list_chunked = [animal_list[i*batch_size:(i+1)*batch_size] for i in range((len(animal_list)+batch_size-1)//batch_size)]

    # save list to JSON file
    json_obj = json.dumps(animal_list_chunked)

    # create jsons directoru
    json_folder = os.path.join(os.getcwd(), "jsons")
    os.makedirs(json_folder, exist_ok=True)

    # create json file name 
    json_file_name = 'animals_n{}_b{}.json'.format(str(len(animal_list)), str(batch_size))

    json_file_path = os.path.join(json_folder, json_file_name)
    with open(json_file_path, "w") as outfile:
        outfile.write(json_obj)
    print("{} created.".format(json_file_path))
    json_file_name = os.path.basename(json_file_path)
    
    num_of_jobs = len(animal_list_chunked)
    
    # number of paths in json
    print('Number of animals in json:', len(animal_list))
    print('Number of chunks with batch_size={}: {}'.format(str(batch_size), str(num_of_jobs)))
    
    # create path to log folder
    log_folder = os.path.join(os.getcwd(), "log")
    os.makedirs(log_folder, exist_ok=True)
    
    # submit job
    subprocess.run(["qsub", 
        "-t", "1-{}".format(str(num_of_jobs)), 
        "-tc", "40", 
        "-o", log_folder,
        "-e", log_folder,
        "behavior_job.sh", 
        json_file_name, #json file path of animal list
        config_file_name#config file path
    ])

    