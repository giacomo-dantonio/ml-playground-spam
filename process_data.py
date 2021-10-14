# * load files and assign labels to them, according to their filepath
# * depending on the hyperparameters process the file content
#     - strip off headers
#     - convert to lowercase
#     - remove puntuaction
#     - replace all URLs with "URL"
#     - replace all numbers with "NUMBER"
#     - perform stemming (trim off word endings)
# * make an ordered set of all words
# * convert data to sparse vectors
# * split training and test set

import argparse
import email
import nltk
import os
import pandas as pd
import re
import typing

from sklearn import model_selection

punctuation_exp = re.compile(r"[\!()\-[\]{};:'\"\\,<>./?@#$%^&*_~]")
url_exp = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
number_exp = re.compile(r"(?:\d{3}[\.,])?\d+(?:[\.,]\d+)?")

LabeledPaths = typing.List[typing.Tuple[str, bool]]
LabeledFiles = typing.Generator[typing.Tuple[str, str, bool], None, None]

def load_files(labeled_paths : LabeledPaths) -> LabeledFiles:
    """
    Load file contents and assign labels to them.
    """
    for (filepath, spam) in labeled_paths:
        with open(filepath, encoding="iso-8859-1") as f:
            yield (os.path.dirname(filepath), f.read(), spam)

def process_file(
    content : str,
    strip_header=False,
    lowercase=False,
    remove_punctuation=False,
    replace_urls=False,
    replace_numbers=False,
    stem=False) -> str:
    """
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
    """

    processed_content = content
    if strip_header:
        msg = email.message_from_string(content)
        processed_content = msg.get_payload()
    
    if lowercase:
        processed_content = processed_content.lower()

    if replace_urls:
        processed_content = url_exp.sub("URL", processed_content)

    if remove_punctuation:
        processed_content = punctuation_exp.sub("", processed_content)

    if replace_numbers:
        processed_content = number_exp.sub("NUMBER", processed_content)

    if stem:
        nltk.download('punkt')
        stemmer = nltk.PorterStemmer()
        tokens = nltk.word_tokenize(processed_content)
        processed_content = " ".join([stemmer.stem(token) for token in tokens])

    return processed_content

def make_dataset(labeled_files : LabeledFiles) -> pd.DataFrame:
    paths, features, labels = zip(*labeled_files)
    return pd.DataFrame({
        "dirpath": paths,
        "mail": features,
        "spam": labels
    })

def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load training data from a file and process them.")

    parser.add_argument(
        "--spam",
        nargs="*",
        default=[],
        help="Path to folders containg spam examples as iso-8859-1 encoded text files.")

    parser.add_argument(
        "--ham",
        nargs="*",
        default=[],
        help="Path to folders containg ham examples as iso-8859-1 encoded text files.")

    parser.add_argument(
        "-s",
        "--strip-headers",
        action="store_true",
        help="Strip the email headers from the training data.")

    parser.add_argument(
        "-l",
        "--lowercase",
        action="store_true",
        help="Convert the email texts to lowercase.")

    parser.add_argument(
        "-r",
        "--remove-punctuation",
        action="store_true",
        help="Remove punctuation from the email content.")

    parser.add_argument(
        "-n",
        "--replace-numbers",
        action="store_true",
        help="Replace number with a placeholder string.")

    parser.add_argument(
        "-u",
        "--replace-urls",
        action="store_true",
        help="Replace urls with a placeholder string.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="data.h5",
        help="A filepath for storing the processed data in the HDF5 format.")

    return parser

def make_labeled_paths(data_dir : str, label : bool):
    for filename in os.listdir(os.path.abspath(data_dir)):
        filepath = os.path.join(data_dir, filename)
        yield (filepath, label)

def process(spam_dirs, ham_dirs, outfile,
    strip_headers=False,
    lowercase=False,
    remove_punctuation=False,
    replace_urls=False,
    replace_numbers=False
):
    labeled_paths = []
    for spam_dir in spam_dirs:
        labeled_paths.extend(make_labeled_paths(spam_dir, True))
    for ham_dir in ham_dirs:
        labeled_paths.extend(make_labeled_paths(ham_dir, False))
 
    labeled_files = [
        (
            dirpath,
            process_file(
                file_content,
                strip_header=strip_headers,
                lowercase=lowercase,
                remove_punctuation=remove_punctuation,
                replace_urls=replace_urls,
                replace_numbers=replace_numbers),
            label
        )
        for (dirpath, file_content, label) in load_files(labeled_paths)
    ]

    ds = make_dataset(labeled_files)

    with pd.HDFStore(outfile) as store:
        split = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        for train_index, test_index in split.split(ds, ds["dirpath"]):
            store["train"] = ds.loc[train_index]
            store["test"] = ds.loc[test_index]

if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    process(
        spam_dirs=args.spam,
        ham_dirs=args.ham,
        outfile=os.path.abspath(args.outpath),
        strip_headers=args.strip_headers,
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        replace_urls=args.replace_urls,
        replace_numbers=args.replace_numbers
    )
