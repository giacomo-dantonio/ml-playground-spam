## Getting started

### Preparing the environment

    virtualenv -p python3 .venv
    source .venv/bin/activate
    pip install nltk pandas sklearn tables

### Generate training and test datasets

    python -m process_data --spam ./data/spam ./data/spam_2 --ham ./data/easy_ham ./data/easy_ham_2 ./data/hard_ham/

### Train model

    (.venv) giacomo@giacomo-ubuntu:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v
    Training a SGD classifier with default hyperparameters.
    Training time: 4.57s
    Metrics computed on the test set
                precision    recall  f1-score   support

        False       1.00      1.00      1.00      1390
            True       0.99      0.99      0.99       480

        accuracy                           1.00      1870
    macro avg       0.99      0.99      0.99      1870
    weighted avg       1.00      1.00      1.00      1870

    Confusion matrix:
            1386    4
            5       475

## Train a model

The module `train_model` train a model for spam classification
using a dataset provided in the HDF5 format.

The HDF5 file should contain two pandas dataframes "train" and "test".
Both should have the same format. I.e.

    Data columns (total 3 columns):
    #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
    0   dirpath  7478 non-null   object
    1   mail     7478 non-null   object
    2   spam     7478 non-null   bool  



    usage: train_model.py [-h] [-g GRIDSEARCH] [-d DATASET] [-o OUTPATH] [-v] [-m]
                        [-c CLASSIFIER]

    Train a model from an HDF5 dataset and export it to a file.

    optional arguments:
    -h, --help            show this help message and exit
    -g GRIDSEARCH, --gridsearch GRIDSEARCH
                            Number of randomized grid search iteration to perform
                            for optimizing the hyperparameters.
    -d DATASET, --dataset DATASET
                            A filepath containing the dataset in the HDF5 format.
    -o OUTPATH, --outpath OUTPATH
                            The filepath for storing the trained model.
    -v, ---verbose        Display information about the trained model on stdout.
    -m, --metrics         Display metrics computed on the test set.
    -c CLASSIFIER, --classifier CLASSIFIER
                            Choose the classifier used for the optimization. The
                            following values are allowed: AdaBoost, Decision Tree,
                            Linear SVM, Nearest Neighbors, Neural Net, Random
                            Forest, RBF SVM, SGD.