# Spam classifier

This is the solution of Exercise 4 of Chapter 3 of the book
*Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow*.

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

The module `train_model` canb be also used as a library.
The following API is exposed.

    NAME
        train_model

    DESCRIPTION
        Train a model for spam classification using pandas dataframe as input.
        The dataframe should have the following format.

            Data columns (total 3 columns):
            #   Column   Non-Null Count  Dtype
            ---  ------   --------------  -----
            0   dirpath  7478 non-null   object
            1   mail     7478 non-null   object
            2   spam     7478 non-null   bool

    FUNCTIONS
        train(data: pandas.core.frame.DataFrame, classifier_name: str = 'SGD', gridsearch: int = None)
            Train a model for spam classification.

            Arguments:

            data            The dataframe used for training the model. It should have
                            the format described above.

            classifier_name The name of the classifier used for building the model.
                            The following classifiers are supported:
                            AdaBoost, Decision Tree, Linear SVM, Nearest Neighbors,
                            Neural Net, Random, Forest, RBF SVM, SGD.

            gridsearch      The number of iterations for the randomized grid search.
                            If you input None, the search will be skipped and the model
                            will be trained using the default hyperparameters
                            of the classifier.

            Returns:
                a tuple (search, model) containg the grid search object
                from scikit learn and the trained model.

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

The module `process_data` canb be also used as a library.
The following API is exposed.

    NAME
        process_data

    DESCRIPTION
        Reads emails from some folders and prepare the train and test datasets
        for the `train_model` module. The folders are also used to assign labels
        (i.e. the files in a folder are either all spam or all ham).

    CLASSES
        sklearn.base.BaseEstimator(builtins.object)
            DataTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)
        sklearn.base.TransformerMixin(builtins.object)
            DataTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)

        class DataTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)
         |  DataTransformer(strip_header=False, lowercase=False, remove_punctuation=False, replace_urls=False, replace_numbers=False, stem=False)
         |
         |  A data transformer to be used in scikit learn pipelines.
         |  It processes emails, given as strings, and offers the same
         |  hyperparameters as `process_file`
         |
         |  Method resolution order:
         |      DataTransformer
         |      sklearn.base.BaseEstimator
         |      sklearn.base.TransformerMixin
         |      builtins.object
         |
         |  Methods defined here:
         |
         |  __init__(self, strip_header=False, lowercase=False, remove_punctuation=False, replace_urls=False, replace_numbers=False, stem=False)
         |         Construct an instance of this transformer.
         |
         |         Parameters:
         |         strip_header (bool): whether the email headers should be
         |                              removed from the file content
         |         lowercase (bool): whether the file content should be converted to lowercase
         |         remove_punctuation (bool): whether the punctuation symbols should be
         |                                    removed from the file content
         |         replace_urls (bool): whether the urls in the file content should be
         |                              replaced by the string "URL"
         |         replace_numbers (bool): whether the numbers in the file content should be
         |                                 replaced by the string "NUMBER"
         |         stem (bool): whether stemming (e.g. trim off word endings) should
         |      .               be perfomed
         |
         |  fit(self, X, y=None)
         |      Since this is a pure transformer, the fit method will not do anything.
         |
         |  transform(self, X, y=None)
         |      Process the content of the series X.
         |
         |      Parameters:
         |      X (pd.Series): A panda series containing the content of the emails
         |                     as strings.
         |      y (pd.Series): The target series. Not used in this implementation.
         |
         |      Returns:
         |      The X series, transformed according to the hyperparameters.
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from sklearn.base.BaseEstimator:
         |
         |  __getstate__(self)
         |
         |  __repr__(self, N_CHAR_MAX=700)
         |      Return repr(self).
         |
         |  __setstate__(self, state)
         |
         |  get_params(self, deep=True)
         |      Get parameters for this estimator.
         |
         |      Parameters
         |      ----------
         |      deep : bool, default=True
         |          If True, will return the parameters for this estimator and
         |          contained subobjects that are estimators.
         |
         |      Returns
         |      -------
         |      params : dict
         |          Parameter names mapped to their values.
         |
         |  set_params(self, **params)
         |      Set the parameters of this estimator.
         |
         |      The method works on simple estimators as well as on nested objects
         |      (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
         |      parameters of the form ``<component>__<parameter>`` so that it's
         |      possible to update each component of a nested object.
         |
         |      Parameters
         |      ----------
         |      **params : dict
         |          Estimator parameters.
         |
         |      Returns
         |      -------
         |      self : estimator instance
         |          Estimator instance.
         |
         |  ----------------------------------------------------------------------
         |  Data descriptors inherited from sklearn.base.BaseEstimator:
         |
         |  __dict__
         |      dictionary for instance variables (if defined)
         |
         |  __weakref__
         |      list of weak references to the object (if defined)
         |
         |  ----------------------------------------------------------------------
         |  Methods inherited from sklearn.base.TransformerMixin:
         |
         |  fit_transform(self, X, y=None, **fit_params)
         |      Fit to data, then transform it.
         |
         |      Fits transformer to `X` and `y` with optional parameters `fit_params`
         |      and returns a transformed version of `X`.
         |
         |      Parameters
         |      ----------
         |      X : array-like of shape (n_samples, n_features)
         |          Input samples.
         |
         |      y :  array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None
         |          Target values (None for unsupervised transformations).
         |
         |      **fit_params : dict
         |          Additional fit parameters.
         |
         |      Returns
         |      -------
         |      X_new : ndarray array of shape (n_samples, n_features_new)
         |          Transformed array.

    FUNCTIONS
        process(spam_dirs, ham_dirs, outfile, strip_headers=False, lowercase=False, remove_punctuation=False, replace_urls=False, replace_numbers=False)
            Process the content of the input folders and write the output datasets
            to an HDF5 archive.

            Parameters:
            spam_dirs ([str]): List of paths to the folders which contain spam.
            ham_dirs ([str]): List of paths to the folders which contain ham.
            outfile (str): Path to the output HDF5 archive.
            strip_header (bool): Whether the email headers should be
                                 removed from the file content.
            lowercase (bool): Whether the file content should be converted to lowercase.
            remove_punctuation (bool): Whether the punctuation symbols should be
                                       removed from the file content.
            replace_urls (bool): Whether the urls in the file content should be
                                 replaced by the string "URL".
            replace_numbers (bool): Whether the numbers in the file content should be
                                    replaced by the string "NUMBER".
            stem (bool): Whether stemming (e.g. trim off word endings) should
                         be perfomed.

        process_file(content: str, strip_header=False, lowercase=False, remove_punctuation=False, replace_urls=False, replace_numbers=False, stem=False) -> str
            Process the file content, by performing the actions specified in the
            following parameters.

            Parameters:
            content (str): the content of the file
            strip_header (bool): whether the email headers should be
                                 removed from the file content
            lowercase (bool): whether the file content should be converted to lowercase
            remove_punctuation (bool): whether the punctuation symbols should be
                                       removed from the file content
            replace_urls (bool): whether the urls in the file content should be
                                 replaced by the string "URL"
            replace_numbers (bool): whether the numbers in the file content should be
                                    replaced by the string "NUMBER"
            stem (bool): whether stemming (e.g. trim off word endings) should
                         be perfomed

            Returns:
            str: the processed file content

    DATA
        LabeledFiles = typing.Generator[typing.Tuple[str, str, bool], NoneType...
        LabeledPaths = typing.List[typing.Tuple[str, bool]]

    FILE
        c:\users\giacomodantonio\documents\ml\spam\process_data.py

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

The module `predict` canb be also used as a library.
The following API is exposed.

    NAME
        predict - Classify emails (spam or ham) using a model trained with the train_model module.

    FUNCTIONS
        predict(texts, model)
            Classify email content using the given model.
            The model must have been built using the module train_model.

            Parameters:
            texts ([str]): The content of the emails to be classified.
            model (object): The model to be used for classification.

            Returns:
            A list of booleans, representing the predictions for the input texts.

    FILE
        c:\users\giacomodantonio\documents\ml\spam\predict.py
