# TODO: grid search for all classifiers
# TODO: replace verbose + print with logging library

import argparse
import pandas as pd
import joblib
import process_data
import utils
import json

from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn import pipeline as pl
from sklearn import svm
from sklearn import tree
from sklearn.feature_extraction import text

classifiers = {
    "AdaBoost": ensemble.AdaBoostClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Linear SVM": svm.SVC(kernel="linear", C=0.025),
    "Nearest Neighbors": neighbors.KNeighborsClassifier(3),
    "Neural Net": neural_network.MLPClassifier(alpha=1, max_iter=1000),
    "Random Forest": ensemble.RandomForestClassifier(),  # third best one
    "RBF SVM": svm.SVC(gamma=2, C=1),  # best one so far
    "SGD": linear_model.SGDClassifier()  # second best one
}

def get_search_parameters(classifier_name):
    if classifier_name == "Decision Tree":
        return {
            "clf__criterion": ("gini", "entropy"),
            "clf__splitter": ("best", "random"),
        }
    elif classifier_name == "SGD":
        return {
            "clf__max_iter": (20, 1000),
            "clf__alpha": (1E-4, 1E-5, 1E-6),
            "clf__penalty": ("l2", "elasticnet"),
            "clf__max_iter": (10, 100, 1000),
        }
    else:
        return {}

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

    parser.add_argument(
        "-c",
        "--classifier",
        default="SGD",
        help="Choose the classifier used for the optimization. "
             "The following values are allowed: %s." % ", ".join(classifiers.keys())
    )

    return parser

def train(data, classifier_name="SGD", gridsearch=None, verbose=False):
    classifier = classifiers.get(classifier_name)

    if classifier is None:
        # FIXME: warning
        classifier_name = "SGD"
        classifier = classifiers.get(classifier_name)

    pipeline = pl.Pipeline([
        ("process", process_data.DataTransformer()),
        ("vect", text.CountVectorizer()),
        ("tfidf", text.TfidfTransformer()),
        ("clf", classifier)
    ])

    search = None
    with utils.Timer() as t:
        if gridsearch is not None:
            if verbose:
                print(
                    "Performing grid search for the preprocessing step "
                    "with a %s classifier." % classifier_name
                )

            parameters = {
                "vect__max_df": (0.5, 0.75, 1.0),
                "vect__max_features": (None, 5000, 10000, 50000),
                "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
                "tfidf__use_idf": (True, False),
                "tfidf__norm": ("l1", "l2"),
                "process__strip_header": (True, False),
                "process__lowercase": (True, False),
                "process__remove_punctuation": (True, False),
                "process__replace_urls": (True, False),
                "process__replace_numbers": (True, False),
            }
            parameters.update(get_search_parameters(classifier_name))

            search = model_selection.RandomizedSearchCV(
                pipeline, parameters,
                n_iter=gridsearch,
                verbose=1,
                scoring="f1",
                cv=model_selection.GroupKFold())

            search.fit(data["mail"], data["spam"], groups=data["dirpath"])
            model = search.best_estimator_
        else:
            if verbose:
                print("Training a %s classifier with default hyperparameters." % classifier_name)
            model = pipeline.fit(data["mail"], data["spam"])

    if verbose:
        print("Training time: {:.2f}s".format(t.elapsed))

    return search, model

def show_metrics(data_filepath, model):
    test = pd.read_hdf(data_filepath, key="test")

    targets = test["spam"]
    predictions = model.predict(test["mail"])
    confusion = metrics.confusion_matrix(targets, predictions)
    report = metrics.classification_report(targets, predictions)

    return (
        "Metrics computed on the test set\n"
        "{report}\n"
        "Confusion matrix:\n{confusion}\n"
    ).format(
        report = report,
        confusion = "\t{0}\t{1}\n\t{2}\t{3}".format(
            confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1])
    )

if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    train_data = pd.read_hdf(args.dataset, key="train")
    search, model = train(train_data, args.classifier, args.gridsearch, args.verbose)

    joblib.dump(model, args.outpath)

    if args.verbose and search is not None:
        print("Search Accuracy:", search.best_score_)
        print("Best parameters:\n", json.dumps(search.best_params_, indent=2))

    if args.metrics:
        print(show_metrics(args.dataset, model))
