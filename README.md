# Spam classifier

This is the solution of Exercise 4 of Chapter 3 of the book
*Hands-On Machine Learning with Scikir-Learn, Keras & Tensorflow*.

It is a spam classifier which can be trained using various models.

## TLDR

### Download the dataset

We use Apache SpamAssassin's public dataset.
Download the files from https://homl.info/spamassassin.
Unzip each file in a separate directory
(e.g. `./data/easy_ham`, `./data/easy_ham_2`, `./data/hard_ham`,
`./data/spam`, `./data/spam_2`).

### Preparing the environment

Create a virtual environment (optional) and install the dependencies.

    virtualenv -p python3 .venv
    source .venv/bin/activate
    pip install nltk pandas sklearn tables

### Generate training and test datasets

Generate the train and test datasets and export them to an HDF5 archive
with the following command.

    $ python -m process_data --spam ./data/spam ./data/spam_2 --ham ./data/easy_ham ./data/easy_ham_2 ./data/hard_ham/

### Train model

Train a model with the following command (see below for more options).

    $ python -m train_model -d ./data.h5 -m -v
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

The module `train_model` trains a model for spam classification
using a dataset provided in the HDF5 format.

The HDF5 file should contain two pandas dataframes "train" and "test".
Both should have the same format. I.e.

    Data columns (total 3 columns):
    #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
    0   dirpath  7478 non-null   object
    1   mail     7478 non-null   object
    2   spam     7478 non-null   bool  

The following options are available.

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

On the SpamAssassin's dataset the most promising models are
SGD, RBF SVM, and Random Forest. SGD is considerably faster than the other two,
so it seems the best choice in this case.

A randomized grid search with 50 iterations did not produce any improvements
on the default parameters.

### API usage

TODO.

## Prepare dataset

The module `process_data` can be used to split training and test data
and serialize them to an HDF5 archive, in the format needed from `train_model`.

The following options are available

    $ python -m process_data -h
    usage: process_data.py [-h] [--spam [SPAM [SPAM ...]]] [--ham [HAM [HAM ...]]]
                        [-s] [-l] [-r] [-n] [-u] [-o OUTPATH]

    Load training data from a file and process them.

    optional arguments:
    -h, --help            show this help message and exit
    --spam [SPAM [SPAM ...]]
                          Path to folders containg spam examples as iso-8859-1  
                          encoded text files.
    --ham [HAM [HAM ...]]
                          Path to folders containg ham examples as iso-8859-1
                          encoded text files.
    -s, --strip-headers   Strip the email headers from the training data.
    -l, --lowercase       Convert the email texts to lowercase.
    -r, --remove-punctuation
                          Remove punctuation from the email content.
    -n, --replace-numbers
                          Replace number with a placeholder string.
    -u, --replace-urls    Replace urls with a placeholder string.
    -o OUTPATH, --outpath OUTPATH
                          A filepath for storing the processed data in the HDF5
                          format.

The data should be organized in different folders. All the files are in a
folder belong to the same class (e.g. either spam or ham). You can specify
multiple folders for the same class.

The parameters `--strip-headers`, `--lowercase`, `--remove-punctuation`,
`--replace-numbers`, `--replace-urls` are also explored in the grid search
of `train_model`. If you intend to use the grid search, you should not
specify them here.

### API usage

TODO.

## Make a prediction

The module `predict` can be used to make predictions on a set of files.
The following options are available:

    $ python -m predict -h
    usage: predict.py [-h] [-m MODEL] [-o OUTPATH] [-v] [paths [paths ...]]

    Classify a mail as spam/non spam, using the specified model.

    positional arguments:
    paths                 Filepaths of the emails to be classified.

    optional arguments:
    -h, --help            show this help message and exit
    -m MODEL, --model MODEL
                          The filepath to the trained model.
    -o OUTPATH, --outpath OUTPATH
                          A path to the output file, containg the predicted
                          values.
    -v, ---verbose        Output the predictions on stdout.

You can provide the paths to the *files* to be predicted in the positional
arguments. If you provide the `--output` argument, the predictions
will be written to a file on disk. If you provide the `--verbose` argument,
they will be printend on stdout.

### API usage

TODO.
