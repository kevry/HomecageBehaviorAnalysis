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
config_file_name=$1

#activate conda environment
module load miniconda
conda activate homecagebehaviorENV

python store_animal_data_helper.py --config_file_name $config_file_name
exit