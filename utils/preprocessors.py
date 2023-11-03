
'''
This file contains helper functions to preprocess the data.
'''

# ===============================================================================
# === Imports ===================================================================

import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import time
import scipy

from utils.io import load_nifti

# ===============================================================================
# === Function to split data into training and test set =========================


def split_data(files, path, k_folds, test_set_size):
    """
    This function is used to generate csv files containing lists of the images
    that belong to the training, validation or test set.
    """
    
    # choose randomly which are used for testing
    rng = np.random.RandomState(97)
    ids = [f.split(".")[0] for f in files]
    test = rng.choice(ids, test_set_size, replace=False)
    train = [f for f in ids if f not in test]
    train_imgs = [f'{path}img/{f}' for f in train]
    train_lbls = [f'{path}lbls/label{f[3:]}' for f in train]

    # generate the k-folds for training and validation
    if k_folds == 0:
        raise Exception("The data must be split into different folds for training and validation \n k_folds needs to be > 1.")
    else:
        # shuffle the array for random alignment
        sequence = np.arange(k_folds)
        indices = np.tile(sequence,len(train))
        indices = indices[0:len(train)]
        np.random.shuffle(indices)

    # generate the paths for the csv files if they do not exist
    if not os.path.exists("../csv_files"):
        os.makedirs("../csv_files")
    
    # write the paths to the csv files
    df = pd.DataFrame(data={'imgs': train_imgs, 'lbls': train_lbls, 'fold': indices})
    df.sort_values(by=['fold']).to_csv('../csv_files/train.csv', index=False)
    test_imgs = [f'{path}img/{f}' for f in test]
    test_lbls = [f'{path}lbls/label{f[3:]}' for f in test]
    pd.DataFrame(data={'imgs': test_imgs, 'lbls': test_lbls}).to_csv('../csv_files/test.csv', index=False)

# ===============================================================================
# === Function to generate 2d slices out of 3d image ============================


def generate_slices(img_files,lbls_files,config):
    """
    This function generates 2d slices out of 3d nifti images.
    """
    # generate slices for the images
    pbar = tqdm(img_files)
    for img in pbar:
        time.sleep(0.025)
        pbar.set_description(f'Processing {img}')
        np_volume = load_nifti(os.path.join(config["data_path"], "img", img))
        slicer(np_volume, img, config, True)

    # generate slices for the lables
    pbar2 = tqdm(lbls_files)
    for lbl in pbar2:
        time.sleep(0.025)
        pbar.set_description(f'Processing {lbl}')
        np_volume2 = load_nifti(os.path.join(config["data_path"], "lbls", lbl))
        slicer(np_volume2, lbl, config, False)

# ===============================================================================
# === Function to generate 2d slices ============================================    


def slicer(np_volume,img,config,is_image):
    """
    Function which is responsible for saving the 2D slices.
    """
    id = img.split(".")[0]
    if is_image:
        if not os.path.exists(os.path.join(config["output_path"], 'img',f"{id}")):
            os.makedirs(os.path.join(config["output_path"], 'img', f"{id}"))
    else:
        if not os.path.exists(os.path.join(config["output_path"], 'lbls',f"{id}")):
            os.makedirs(os.path.join(config["output_path"], 'lbls', f"{id}"))

    # save the 2d slices
    z_dim = np_volume.shape[2]
    pbar = tqdm(range(0,z_dim))
    for z in pbar:          
        pbar.set_description(f'Processing slices {z}/{z_dim}')
        z_slice = np_volume[:,:,z]
        if is_image:
            z_slice_norm = normalize2(z_slice)
            cv2.imwrite(os.path.join(config["output_path"], "img", f"{id}", f"{id}_{z:04}.tif"), z_slice_norm)
        else:
            z_slice = z_slice.astype(np.uint16)
            cv2.imwrite(os.path.join(config["output_path"], "lbls", f"{id}", f"{id}_{z:04}.tif"), z_slice)
    print(f"Saved {z_dim} slices for {img}.")

# ===============================================================================
# === Function to normalize the images ==========================================


def normalize(volume):
    """
    Takes numpy volume as input and normalizes it (0-255).
    """
    volume = volume.astype(np.float32)
    volume = volume - np.mean(volume)
    volume = volume / np.std(volume)
    return volume
    
def normalize2(volume):
    """
    Takes numpy volume as input and normalizes it (0-1).
    """
    volume = volume.astype(np.float32)
    mini = np.min(volume)
    maxi = np.max(volume)
    volume = (volume - mini) / (maxi - mini)
    volume = volume.astype("float32")

    return volume

# ===============================================================================
# === Function to resize a numpy volume =========================================


def resize_volume(volume,x,y):
    """
    Takes in a numpy volume and resizes it.
    """
    desired_width = x
    desired_height = y
    # Get current dimensions
    current_width = volume.shape[0]
    current_height = volume.shape[1]
    # Compute factors
    width = current_width / desired_width
    height = current_height / desired_height
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize the image
    img = scipy.ndimage.zoom(volume, (width_factor, height_factor), order=0)

    return img

# ===============================================================================
# === Function to resize a numpy volume =========================================


def resize_volume_3d(volume,x,y,z):
    """
    Takes in a numpy volume and resizes it.
    """
    desired_width = x
    desired_height = y
    desired_depth = z
    # Get current dimensions
    current_width = volume.shape[0]
    current_height = volume.shape[1]
    current_depth = volume.shape[2]
    # Compute factors
    width = current_width / desired_width
    height = current_height / desired_height
    depth = current_depth / desired_depth
    width_factor = 1 / width
    height_factor = 1 / height
    depth_factor = 1 / depth
    # Resize the image
    img = scipy.ndimage.zoom(volume, (width_factor, height_factor, depth_factor), order=1)

    return img

# ===============================================================================
# === Function to preprocess a scan =============================================


def process_scan(path, x, y):
    """
    Read in nifti file and resize the numpy volume.
    """
    # Read scan
    volume = cv2.imread(path,2).astype(np.float32)
    # Resize width, height and depth
    volume = resize_volume(volume,x,y)
    # normalize
    # volume = normalize(volume)

    return volume

# ===============================================================================
# === Function to preprocess an annotation ======================================


def process_annotation(path, x, y):
    """
    Read in nifti file and resize the numpy volume.
    """
    # Read scan
    volume = cv2.imread(path,2).astype(np.uint16)
    # Resize width, height and depth
    volume = resize_volume(volume, x, y)

    return volume

