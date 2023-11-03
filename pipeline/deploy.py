
'''
This file is for loading a model and perform predictions.
'''

# ===============================================================================
# === Imports ===================================================================
import nibabel as nib
import tensorflow as tf
import argparse
import json
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
import nibabel as nib

import sys
sys.path.append("..")
from utils.predict import load_data_paths
from utils.preprocessors import process_scan
from utils.io import load_nifti
from utils.io import save_nifti
from utils.preprocessors import resize_volume

# ===============================================================================
# === Start prediction ==========================================================


def start_prediction(deploy_config):
    """
    This function starts the prediction on the test scan.
    """

    # load the model
    print("Loading stored model...\n")
    model = tf.keras.models.load_model(deploy_config["path_to_model"],compile=False)

    # generate paths to test data
    img_paths, lbl_paths = load_data_paths(deploy_config)
    img_name = img_paths[0].split("/")[-1].split(".")[0].split("_")[0]
    # load the data according to the paths
    print("Loading imgage slices...")
    try:
        y_original = nib.load(f'../data/lbls/{lbl_paths[0].split("/")[3]}{deploy_config["file_ending"]}')
        x_test = np.array([process_scan(path,deploy_config["x"],deploy_config["y"]) for path in tqdm(img_paths)])
        y_test = np.array([load_nifti(f'../data/lbls/{lbl_paths[0].split("/")[3]}{deploy_config["file_ending"]}')])
        print("Data was loaded successfully.\n")
    except:
        raise Exception('An error occured while loading the data.\n')

    # predict slices
    print("Start of prediction...\n")
    predicted = model.predict(x_test)
    predicted_arg = np.argmax(predicted,-1)
    list_of_slices = list(predicted_arg)
    
    print("Resizing prediction...\n")
    # extract original size of image
    resized_lbl = y_test[0]
    x,y,_ = np.shape(resized_lbl)
    list_of_resized_slices = [resize_volume(i,x,y) for i in list_of_slices]

    print("Stacking slices to 3D image...\n")
    resized_mask = np.dstack(list_of_resized_slices)

    # create and save nifti file
    print("Saving prediction as NIfTI file...\n")
    save_nifti(resized_mask,y_original,f"pred_{img_name}")

# ===============================================================================
# === Generate an argument parser ===============================================


if __name__ == '__main__':
    """
    Generate argument parser to get the configuration for prediction.
    """
    parser = argparse.ArgumentParser(description='Prediction script')
    parser.add_argument('--config_file', '-c', default='../config/deploy_config.json')
    args = parser.parse_args()
    with open(args.config_file) as f:
        deploy_config = json.load(f)

    # start the prediction
    start_prediction(deploy_config)

