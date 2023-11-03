
'''
This file contains the code to run the training procedure
on the preprocessed data.
'''

# ===============================================================================
# === Imports ===================================================================

import argparse
import tensorflow as tf
import json
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("..")
from utils.training import load_model
from utils.training import gen_optimizer
from utils.training import load_data_paths
from utils.preprocessors import process_scan
from utils.preprocessors import process_annotation
from utils.metrics import MeanIoU
from utils.augmentations import Augment

# ===============================================================================
# === REMARK ====================================================================

"""
Before running the training script (pipeline/train.py) the data preprocessing script
(pipeline/preprocessing.py) should be run first in order to prepare the data
correctly for the training.
"""

# ===============================================================================
# === Function to start training ================================================

def start_training_session(train_config):
    """
    This function starts the training procedure.
    """

    # load the data paths
    img_paths_train, lbl_paths_train, img_paths_val, lbl_paths_val = load_data_paths(train_config)
    print("Checking if the correct paths were found:")
    if len(img_paths_train) == len(lbl_paths_train) and len(img_paths_val) == len(lbl_paths_val):
        print("The same number of images and labels was found.")
        print(f'Training scans: {len(img_paths_train)}')
        print(f'Validation scans: {len(img_paths_val)}')
    else:
        raise Exception("Wrong number of images or labels was found. \n")
    
    # load the data according to the paths
    try:
        print("Loading the training scans...")
        x_train = np.array([process_scan(path,train_config["x"],train_config["y"]) for path in tqdm(img_paths_train)])
        print("Loading the training annotations...")
        y_train = np.array([process_annotation(path,train_config["x"],train_config["y"]) for path in tqdm(lbl_paths_train)])

        # compute class weights if defined in config file
        if train_config["compute_class_weights"] == 1:
            print("Computing class weights... \n")
            n_samples = len(y_train) * train_config["x"] * train_config["y"]
            n_classes = train_config["out_channels"]
            binc = np.bincount(y_train.flatten())
            class_weights = n_samples / (n_classes * binc)
        else:
            class_weights = np.ones(7)
            print("Class weights are assumed to be equal...\n")
        class_weights_print = dict(zip([i for i in range(train_config["out_channels"])], class_weights))
        print(f"Computed class weights: {class_weights_print} \n")
        
        # load the validation scans
        print("Loading the validation scans...")
        x_val = np.array([process_scan(path,train_config["x"],train_config["y"]) for path in tqdm(img_paths_val)])
        print("Loading the validation annotations...")
        y_val = np.array([process_annotation(path,train_config["x"],train_config["y"]) for path in tqdm(lbl_paths_val)])

        print("Data was loaded successfully. \n")
    except:
        raise Exception('An error occured while loading the data. \n')
    
    # define the tensorflow data loaders
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # generate the tensorflow datasets and generate batches according to the batch size defined in config file.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .batch(train_config["batch_size"])
        .prefetch(2)
    )
    val_dataset = (
        val_loader.shuffle(len(x_val))
        .batch(train_config["batch_size"])
        .prefetch(2)
    )

    # load the model which is specified in the config file.
    # other models should be implemented in the ../utils/networks.py file and in the ../utils/training.py file.
    model = load_model(train_config)                                       

    # define the specified optimizer and the learing rate sheduler
    # other optimizers have to be implemented in the ../utils/training.py file and in the ../utils/training.py file.
    optimizer, scheduler_cb = gen_optimizer(train_config)

    # defined the loss function and metric
    # other loss functions and metrics have to be implemented here and in the ../utils/metrics.py file, so they can be specified in the config file.
    if train_config["loss"] == "scce":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        raise Exception("Loss choosen in config is not implemented. \n")

    if train_config["metric"] == "meaniou":
        metric = MeanIoU(num_classes=train_config["out_channels"])
    elif train_config["metric"] == "sca":
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
    else:
        raise Exception("Metric choosen in config is not implemented. \n")

    # compile the model
    model.compile(loss = loss,
            metrics=[metric],
            optimizer=optimizer)

    # define checkpoint
    checkpoint_filepath = f'../{train_config["save_path"]}/{train_config["name"]}.h5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                        verbose=1,
                                        save_best_only=True,
                                        monitor="val_mean_io_u",
                                        mode='max')

    # set up tensorboard
    log_dir = f'../{train_config["save_path"]}/logs/'
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks = [checkpoint_cb, tensorboard_cb, scheduler_cb]

    # data augmentations
    # other data augmentations have to be specifed in the ../utils/augmentations.py file.
    if train_config["augmentations"] != "":
        print("Applying data augmentations... \n")
        train_dataset = train_dataset.map(Augment(train_config))

    # function for creating class weights
    def create_class_weights(image, label, class_weights=class_weights):
        """
        This function creates class weights in order to balance the classes.
        """
        # Create an image of `sample_weights` by using the label at each pixel as an 
        # index into the `class weights` .
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int64))

        return image, label, sample_weights

    ######## TRAIN THE MODEL ########
    print("START OF TRAINING \n")
    model.fit(
        train_dataset.map(create_class_weights),
        epochs=train_config["epochs"],
        validation_data=val_dataset,
        shuffle = True,
        callbacks=callbacks,
        verbose=1)

# ===============================================================================
# === Generate an argument parser ===============================================

if __name__ == '__main__':
    """
    Generate argument parser to get the configuration for training on the
    preprocessed data and check if a GPU device is available for tensorflow.
    """
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config_file', '-c', default='../config/train_config.json')
    args = parser.parse_args()
    with open(args.config_file) as f:
        train_config = json.load(f)
    
    # set up the gpu for tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Tensorflow detected GPU.")
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    # start training on the data
    start_training_session(train_config)

