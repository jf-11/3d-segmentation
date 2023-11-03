
'''
This file contains helper functions to train on the data.
'''

# ===============================================================================
# === Imports ===================================================================

import tensorflow as tf
import pandas as pd
import os

from utils.networks import build_unet

# ===============================================================================
# === This function loads the specified model ===================================

# parameters for models: x, y, in_channels, out_channels
def load_model(train_config):
    """
    Helper function to load a model architecture accoring to the config file.
    """
    if train_config["model"] == "unet1024":
        model = build_unet((train_config["x"], train_config["y"], train_config["in_channels"]), train_config["out_channels"])
    else:
        raise Exception("The model specified in the configuration file is not implemented.")
    return model

# ===============================================================================
# === This function returns the optimizer  ======================================

def gen_optimizer(train_config):
    """
    Helper function to load the specified optimizer and learning rate sheduler.
    """
    if train_config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=train_config["learning_rate"])
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                            patience=5, min_lr=train_config["min_lr"], verbose=1)
    else:
        raise Exception("The optimizer specified in the configuration is not implemented.")
    return optimizer, reduce_lr

# ===============================================================================
# === Function to load the data paths ===========================================

def load_data_paths(train_config):
    """
    Function to load the datapaths according to the csv and config file.
    """
    # extract the path to the csv file
    path_train_file = train_config["csv_files"] + "/train.csv"

    # divide the paths into training and validation set according to val_set_index in config
    file_paths = pd.read_csv(path_train_file, dtype=object, keep_default_na=False, na_values=[]).values
    train_set = [i for i in file_paths if int("".join(i[2]))!=train_config["val_set_index"]]
    val_set = [i for i in file_paths if int("".join(i[2]))==train_config["val_set_index"]]

    # complete the path to the images and labels
    # training sets
    img_paths_train = []
    img_paths_train += [train_set[img_path][0] + "/" + listed_dir for img_path in range(0,len(train_set)) for listed_dir in sorted(os.listdir(train_set[img_path][0])) if not listed_dir.startswith('.')]
    lbl_paths_train = []
    lbl_paths_train += [train_set[img_path][1] + "/" + listed_dir for img_path in range(0,len(train_set)) for listed_dir in sorted(os.listdir(train_set[img_path][1])) if not listed_dir.startswith('.')]

    # validation sets
    img_paths_val = []
    img_paths_val += [val_set[img_path][0] + "/" + listed_dir for img_path in range(0,len(val_set)) for listed_dir in sorted(os.listdir(val_set[img_path][0])) if not listed_dir.startswith('.')]
    lbl_paths_val = []
    lbl_paths_val += [val_set[img_path][1] + "/" + listed_dir for img_path in range(0,len(val_set)) for listed_dir in sorted(os.listdir(val_set[img_path][1])) if not listed_dir.startswith('.')]

    return img_paths_train, lbl_paths_train, img_paths_val, lbl_paths_val

