# 3D Segmentation of organs using deep learning

<p>
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
    <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/>
</p>

<img width="274" alt="logo" src="logo.png">

## About
This repository contains a pipeline for medical image segmentation. The `pipeline` folder consists out of three core modules. The file `preprocessing.py` takes care of the preparation of the training data. It takes 3D images and slices them into 2D slices. Further two *csv files* are generated containing paths to training, validation and test data.
The file `train.py` is used for the training of the data. It trains and saves a model under the specified path.
The file `deploy.py` if used for inference. It loads the model from the specifed path and predicts the segmentation map on the test image. The segmentation map is saved as a 3D `NIfTI` image.
The `config` folder contains `.json` config files for these three main modules and several settings/ paths can be specified there.

The implemented model is a Unet inspired by [Olaf Ronneberger, Philipp Fischer, and Thomas Brox](https://arxiv.org/pdf/1505.04597.pdf). Implemented as Loss function is the built in Sparse Categorical Crossentropy from tensorflow. Implemented metrics cover the built in metrics Sparse Categorical Accuracy and MeanIoU from tensorflow.

## Required packages

<img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/><img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white"/><img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white"/><img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/><img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/><img src="https://img.shields.io/badge/json-f0dd67?style=for-the-badge&logo=json&logoColor=black"/><img src="https://img.shields.io/badge/tqdm-0998eb?style=for-the-badge&logo=tqdm"/<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black"/><img src="https://img.shields.io/badge/Nibabel-45dfed?style=for-the-badge"/><img src="https://img.shields.io/badge/os-677075?style=for-the-badge"/><img src="https://img.shields.io/badge/sys-3d4042?style=for-the-badge"/><img src="https://img.shields.io/badge/time-6e5258?style=for-the-badge"/>

## Getting started

### Installation

Clone the repo:
```
git clone https://https://github.com/jf-11/3d-segmentation.git
```

### Prerequisites

- Place your folder with the data into the root folder (should be named `data`)
- `data` folder should contain subfolders named `img` and `lbls`
- Naming convention for images: `img0000.nii.gz - imgXXXX.nii.gz`
- Naming convention for labels: `label0000.nii.gz - labelXXXX.nii.gz`

## Usage
    
### Data Preprocessing

Once the data is placed according to the guidelines in Prerequisites, the data is ready to be preprocessed. The `preprocessing.py` modulue slices the 3D images into $z$ 2D slices and stores them in a folder `preprocessed_data` as `.tif` files. In the `preprocessing_config.json` file you can specify an alternative path to your data (internal structure should be as described above) and you can specify an alternative path for storing the preprocessed data. You can specify into how many folds, you want to divide your data (later used for training and validation set).

Run the preprocessing:

```
python preprocessing.py
```

or `python preprocessing.py -c PATH_TO_PREPROCESSING_CONFIG` if the path to the config file differs from the default.

Check if the preprocessing worked. A new folder in the specified output path should have been created.
The new folder should contain an `img` and a `lbls` folder. These folders should contain subfolders named `img0000 - imgXXXX`
and `label0000 - labelXXXX`. These folders should contain the $z$ 2D slices in name format `imgXXXX_XXXX.tif` and
`labelXXXX_XXXX.tif`. A folder named `csv_files` should be generated as well containing the files `train.csv` and `test.csv`. In the files the subfolders should be listed in the format: `../preprocessed_data/img/img0000`, `../preprocessed_data/lbls/label0000` with a third column indicating the fold for k-fold crossvalidation.

### Training

In the `train_config.json` file you can specify a name (the model is saved according to this name) for your training run. Further with `out_channels` you can set the number of your output classes. With `x` and `y` you can define the size to which the images should be resized. You can choose the loss function, the metrics, batch size and number of epochs. In data augmentations you can specify the augmentations that should be applied to your training data. With `min_lr` you can set the minimum learning rate for the learning rate sheduler.

Run the training:

```
python train.py
```

or `python train.py -c PATH_TO_TRAIN_CONFIG` if the path to the config file differs from the default.

You can monitor the loss and the accuracy during training using tensorboard. 
To run tensorboard type:

```
tensorboard --logdir="PATH_TO_LOG_FOLDER"
```

Check if the training of the model worked. Under the specified path you should find a log folder and a 
folder called like the name of your training run specified in the config.

### Deploy

The path to the model you want to load should point to the `run_name.h5` model and should be specified with `path_to_model`. With `x` and `y` you have to define the size of the images the model was trained on. The images are later resized to the original size according to their original label.

Run the prediction:

```
python deploy.py
```

or `python deploy.py -c PATH_TO_DEPLOY_CONFIG` if the path to the config differs from the default.

You should now find a new folder called `predictions`. This folder should contain the predicted image with the name `image_name.nii.gz`.

