# * import dataset from hdf5 file
# * apply pipeline with CountVectorizer, TfidTransformer and some classifier
# * hyperparameter to choose the classifier
# * perform grid search on all the hyperparameters
# * serialize the trained model to a file (use joblib)

# TODO: find better estimator
# TODO: use stratified k-fold for the grid search
# TODO: use several classificators
# TODO: replace verbose + print with logging library

import argparse
import pandas as pd
import joblib

from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import pipeline as pl
from sklearn.feature_extraction import text

def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a model from an HDF5 dataset and export it to a file.")

    parser.add_argument(
        "-g",
        "--gridsearch",
        type=int,
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

    parser.add_argument(
        "-v",
        "---verbose",
        action="store_true",
        help="Display information about the grid search for hyperparameters tuning."
    )

    parser.add_argument(
        "-m",
        "--metrics",
        action="store_true",
        help="Display metrics computed on the test set."
    )

    return parser

def train(data_filepath, outpath, gridsearch=None, verbose=False):
    train = pd.read_hdf(data_filepath, key="train")

    pipeline = pl.Pipeline([
        ('vect', text.CountVectorizer()),
        ('tfidf', text.TfidfTransformer()),
        ('clf', linear_model.SGDClassifier())
    ])

    search = None
    if gridsearch is not None:
        if verbose:
            print("Performing grid search with a linear SGD classifier.")

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
            pipeline, parameters, n_iter=gridsearch, verbose=1, scoring="f1")

        search.fit(train["mail"], train["spam"])
        model = search.best_estimator_
    else:
        if verbose:
            print("Training a linear SGD classifier with default hyperparameters.")
        model = pipeline.fit(train["mail"], train["spam"])

    joblib.dump(model, outpath)

    return search, model

def compute_metrics(data_filepath, model):
    test = pd.read_hdf(data_filepath, key="test")

    targets = test["spam"]
    predictions = model.predict(test["mail"])

    confusion = metrics.confusion_matrix(targets, predictions)
    precision = metrics.precision_score(targets, predictions)
    recall = metrics.recall_score(targets, predictions)
    f1 = metrics.f1_score(targets, predictions)

    return {
        "confusion_matrix": confusion,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    search, model = train(args.dataset, args.outpath, args.gridsearch, args.verbose)

    if args.verbose and search is not None:
        print("Search Accuracy:", search.best_score_)
        print("Best parameters:", search.best_params_)

    if args.metrics:
        values = compute_metrics(args.dataset, model)
        cm = values["confusion_matrix"]

        print("Metrics computed on the test set")
        print("confusion matrix:\n\t{0}\t{1}\n\t{2}\t{3}"
            .format(cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))
        print("precision:", values["precision"])
        print("recall:", values["recall"])
        print("f1 score:", values["f1"])
