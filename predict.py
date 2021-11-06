"""
Classify emails (spam or ham) using a model trained with the train_model module.
"""

import argparse
import joblib
import pandas as pd

def predict(texts, model):
    """
    Classify email content using the given model.
    The model must have been built using the module train_model.

    Parameters:
    texts ([str]): The content of the emails to be classified.
    model (object): The model to be used for classification.

    Returns:
    A list of booleans, representing the predictions for the input texts.
    """
    if isinstance(texts, str):
        texts = [texts]

    data = pd.Series(texts)
    predictions = model.predict(data)
    return predictions.tolist()

def _load_files(filepaths):
    for filepath in filepaths:
        with open(filepath, encoding="iso-8859-1") as f:
            yield f.read()

def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a mail as spam/non spam, using the specified model.")

    parser.add_argument(
        "paths",
        nargs="*",
        help="Filepaths of the emails to be classified.")

    parser.add_argument(
        "-m",
        "--model",
        default="model.pkl",
        help="The filepath to the trained model.")

    parser.add_argument(
        "-o",
        "--outpath",
        help="A path to the output file, containg the predicted values.")

    parser.add_argument(
        "-v",
        "---verbose",
        action="store_true",
        help="Output the predictions on stdout."
    )

    return parser

def _show_predictions(paths, predictions):
    justify = max([len(path) for path in paths])

    lines = []
    for filepath, prediction in zip(paths, predictions):
        lines.append("%s  %s" % (filepath.ljust(justify), prediction))

    return "\n".join(lines)

if __name__ == "__main__":
    parser = _make_argparser()
    args = parser.parse_args()

    model = joblib.load(args.model)
    texts = list(_load_files(args.paths))

    predictions = predict(texts, model)
    output = _show_predictions(args.paths, predictions)

    if args.verbose:
        print(output)

    if args.outpath:
        with open(args.outpath, "w") as f:
            f.write(output)
