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
