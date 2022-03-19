# Phonon vibration frequency prediction using deeperGATGNN graph neural network

This software package implements our developed model deeperGATGNN for vibrational frequency prediction.

[Machine Learning and Evolution Laboratory](http://mleg.cse.sc.edu)
Department of Computer Science and Engineering
University of South Carolina

# Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)

# Introduction
This package provides 3 major functions:
* Train a deeperGATGNN model for predicting vibrational frequencies.
* Evaluating performance of the trained deeperGATGNN model.
* Predict the property of a given material using its POSCAR file.

The following paper provides more details about the framework and use: https://arxiv.org/abs/2111.05885

# Installation
* Pytorch (tested on 1.8.1)
* Numpy (tested on 1.20.3)
* Pandas (tested on 1.2.4)
* Scikit-learn (tested on 0.24.2)
* PyTorch-Geometric (tested on 1.7.0)
* Pymatgen (tested on 2022.0.8)

```
pip install torch torchvision
pip install numpy
pip install pandas
pip install scikit-learn
pip install torch-geometric
pip install pymatgen
```

# Dataset

### Training Directory Format
Inside the training directory, there should be a directory for each material id in the dataset. Each POSCAR and OUTCAR should be in a directory with their respective IDs. The POSCAR file should be named "POSCAR", and the OUTCAR file should be named "OUTCAR".

### Prediction Directory Format
The prediction directory should be formatted similarly to the training data set. Inside the prediction directory, there should be a directory for each material id in the dataset. Each material's POSCAR should be in its respective directory. The POSCAR files should be titled "POSCAR". There is no need to also include OUTCAR files in the prediction directory.

# Usage
This package can be used to train, evaluate, and predict vibrational frequencies. It is important to be consistent when changing the model's parameters across training, evaluating, and predicting.

### Training a new model
* Example 1: Train a model with data set in the "DATA/POSCAR-data/" directory. Running this command will also extract all the vibrational frequencies from the data set's OUTCAR files and write it to the file, "DATA/properties-reference/vibrationfrequency.csv"
```
python train.py
```
* Example 2: Train a model with data set in the "DATA/POSCAR-data/" directory without extracting the vibrational frequences from the data set's OUTCAR files. Use this flag if the "DATA/properties-reference/vibrationfrequency.csv" already exists as it will save time.
```
python train.py --create_prop_ref False  # "false" will also work
```
* Example 3: Train a model with a data set with a specified path to a training directory. Use the --indir flag to specify the filepath.
```
python train.py --indir some_directory_path
```

### Evaluating the model
* To evaluate the model, run this command.
```
python evaluate.py
```

### Predicting the model
* Example 1: Predict a model with prediction directory in "DATA/prediction-directory/"
```
python predict.py
```
* Example 2: Predict a model with a specified path to a prediction directory.
```
python predict.py --dir some_directory_path
```
