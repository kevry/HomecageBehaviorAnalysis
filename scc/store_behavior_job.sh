#!/bin/bash -l

#this merges output and error files into one file
#$ -j y

#this sets the project for the script to be run under
#$ -P jchenlab

#number of cores to select
#$ -pe omp 16

#specify the time limit
#$ -l h_rt=12:00:00

#activate conda environment
module load miniconda
conda activate homecagebehaviorENV

slptime=$(echo "scale=4 ; ($RANDOM/32768) * 10" | bc)
sleep $slptime

#json file path
json_file_name=$1

#configuration file path
config_file_name=$2

#run main python script
cd ..
python store_behavior_data_scc.py --json_file_name $json_file_name --config_file_name $config_file_name
exit
