
'''
This file contains helper functions to make predictions.
'''

# ===============================================================================
# === Imports ===================================================================

import pandas as pd
import os
import numpy as np

# ===============================================================================
# === Function to load the data paths ===========================================

def load_data_paths(deploy_config):
    """
    Function which loads the datapaths according to the config file.
    """
    # extract the path to the csv file
    file_path = deploy_config["csv_files"] + "/test.csv"

    # divide the paths into training and validation set according to val_set_index in config
    file_paths = pd.read_csv(file_path, dtype=object, keep_default_na=False, na_values=[]).values
    img_paths = [i[0] for i in file_paths]
    lbl_paths = [i[1] for i in file_paths]

    # complete the path to the images and labels
    # training sets
    paths_to_img = []
    paths_to_img += [img_paths[img_path] + "/" + listed_dir for img_path in range(0,len(img_paths)) for listed_dir in sorted(os.listdir(img_paths[img_path])) if not listed_dir.startswith('.')]

    paths_to_lbl = []
    paths_to_lbl += [lbl_paths[lbl_path] + "/" + listed_dir for lbl_path in range(0,len(lbl_paths)) for listed_dir in sorted(os.listdir(lbl_paths[lbl_path])) if not listed_dir.startswith('.')]

    return paths_to_img, paths_to_lbl
