#!/bin/bash -l

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

#number of cores to select
#$ -pe omp 4

#specify the time limit
#$ -l h_rt=12:00:00

#path to configuration file
config_file_path=$1

module load python3
python collect_animal_data_helper.py --config_file_path config_file_path
exit