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

punctuation_exp = re.compile(r"[\!()\-[\]{};:'\"\\,<>./?@#$%^&*_~]")
url_exp = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
number_exp = re.compile(r"(?:\d{3}[\.,])?\d+(?:[\.,]\d+)?")

LabeledPaths = typing.List[typing.Tuple[str, bool]]
LabeledFiles = typing.Generator[typing.Tuple[str, bool], None, None]

def load_files(labeled_paths : LabeledPaths) -> LabeledFiles:
    """
    Load file contents and assign labels to them.
    """
    for (filepath, spam) in labeled_paths:
        with open(filepath, encoding="iso-8859-1") as f:
            yield (f.read(), spam)

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
    features, labels = zip(*labeled_files)
    return pd.DataFrame({
        "mails": features,
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
        action="store_false",
        help="Strip the email headers from the training data.")

    parser.add_argument(
        "-l",
        "--lowercase",
        action="store_false",
        help="Convert the email texts to lowercase.")

    parser.add_argument(
        "-o",
        "--outpath",
        default="train.data",
        help="A filepath for storing the processed data in an internal format.")
        
    return parser

def make_labeled_paths(data_dir : str, label : bool) -> LabeledFiles:
    for filename in os.listdir(os.path.abspath(data_dir)):
        filepath = os.path.join(data_dir, filename)
        yield (filepath, label)

if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    labeled_paths = []
    for spam_dir in args.spam:
        labeled_paths.extend(make_labeled_paths(spam_dir, True))
    for ham_dir in args.ham:
        labeled_paths.extend(make_labeled_paths(ham_dir, False))
 
    labeled_files = load_files(labeled_paths)

    print(labeled_paths)