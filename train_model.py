# * import dataset from hdf5 file
# * apply pipeline with CountVectorizer, TfidTransformer and some classifier
# * hyperparameter to choose the classifier
# * perform grid search on all the hyperparameters
# * serialize the trained model to a file (use joblib)

# TODO: find better estimator
# TODO: use several classificators
# TODO: output the scoring and the best parameters (use a -v --verbose flag for this)

import argparse
from numpy import mod
import pandas as pd
import joblib

from sklearn import model_selection
from sklearn import linear_model
from sklearn import pipeline as pl
from sklearn.feature_extraction import text

def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a model from an HDF5 dataset and export it to a file.")

    parser.add_argument(
        "-g",
        "--gridsearch",
        type=int,
        default=10,
        help="Number of randomized grid search iteration to perform for optimizing the hyperparameters.")

    parser.add_argument(
        "-d",
        "--dataset",
        help="A filepath containing the dataset in the HDF5 format.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="model.pkl",
        help="The filepath for storing the trained model.")

    return parser

def train(data_filepath, outpath, gridsearch=10):
    train = pd.read_hdf(data_filepath, key="train")
    # train = dataset["train"]

    pipeline = pl.Pipeline([
        ('vect', text.CountVectorizer()),
        ('tfidf', text.TfidfTransformer()),
        ('clf', linear_model.SGDClassifier())
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': (20,),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }

    search = model_selection.RandomizedSearchCV(
        pipeline, parameters, n_iter=gridsearch, verbose=1)

    search.fit(train["mail"], train["spam"])
    model = search.best_estimator_

    joblib.dump(model, outpath)

if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    train(args.dataset, args.outpath, args.gridsearch)
