
'''
This file contains helper function to load images.
'''

# ===============================================================================
# === Imports ===================================================================

import nibabel as nib
import numpy as np
import os

# ===============================================================================
# === Function to read in Nifti files ===========================================

def load_nifti(path_to_file):
    '''
    This function takes the Nifti file under the specified path as input and returns
    a numpy volume.
    The whole path of the images has to be specified.
    '''
    scan = nib.load(path_to_file)
    scan = scan.get_fdata()
    return scan

def save_nifti(mask,gt,name):
    """
    This function takes a numpy volume and saves it as a prediction.
    """
    mask = mask.astype(np.int16)
    img = nib.Nifti1Image(mask, gt.affine, gt.header)
    nib.save(img, f"../predictions/{name}.nii.gz")
    print("Saved prediction as Nifti file.")
