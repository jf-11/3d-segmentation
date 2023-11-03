
'''
This file contains the code to preprocess the data in order
to run the training.
'''

# ===============================================================================
# === Imports ===================================================================

import os
import argparse
import json

import sys
sys.path.append("..")
from utils.preprocessors import split_data
from utils.preprocessors import generate_slices

# ===============================================================================
# === Function to call the preprocessing ========================================


def preprocess(preprocessing_config):
    """
    This function preprocesses the data with the help of the helper functions
    which are defined in the utils folder.
    """
    # get all files in the img directory of the data
    img_files = [img for img in sorted(os.listdir(os.path.join(preprocessing_config["data_path"],"img"))) if not img.startswith('.')]
    lbls_files = [lbl for lbl in sorted(os.listdir(os.path.join(preprocessing_config["data_path"],"lbls"))) if not lbl.startswith('.')]
    
    # call the function split_data to split the data into training and test set
    split_data(img_files, preprocessing_config["output_path"], preprocessing_config["k_folds"], preprocessing_config["test_set_size"])
    print("CSV file generated and saved. \n")

    # generate path for output if it doesn't exist
    if not os.path.exists(os.path.join(preprocessing_config["output_path"], 'img')):
        os.makedirs(os.path.join(preprocessing_config["output_path"], 'img'))

    # generate path for the labels if it doesn't exist
    if not os.path.exists(os.path.join(preprocessing_config["output_path"], 'lbls')):
        os.makedirs(os.path.join(preprocessing_config["output_path"], 'lbls'))

    # call the function generate_slices in order to generate the 2d slices
    generate_slices(img_files,lbls_files,preprocessing_config)

    # printing to the console
    print(f'Data prepared for training and saved in the folder {preprocessing_config["output_path"]}. \n')

# ===============================================================================
# === Generate an argument parser ===============================================


if __name__ == '__main__':
    """
    Generate argument parser to get the configuration for preprocessing from
    the preprocessing_config.json file and run the preprocessing.
    """
    parser = argparse.ArgumentParser(description='CT preprocessing script')
    parser.add_argument('--config_file', '-c', default='../config/preprocessing_config.json')
    args = parser.parse_args()
    with open(args.config_file) as f:
        preprocessing_config = json.load(f)

    # Start preprocessing of the data
    preprocess(preprocessing_config)

