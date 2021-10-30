# Comparison of the different classifiers

## AdaBoost

    $ python -m train_model -d ./data.h5 -m -v -c "AdaBoost"
    INFO  Training a AdaBoost classifier with default hyperparameters.
    INFO  Training time: 16.95s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.99      0.99      0.99      1390
            True       0.98      0.98      0.98       480

        accuracy                           0.99      1870
       macro avg       0.99      0.99      0.99      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1380  10
      8     472

## Decision Tree

    $ python -m train_model -d ./data.h5 -m -v -c "Decision Tree"
    INFO  Training a Decision Tree classifier with default hyperparameters.
    INFO  Training time: 8.81s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.97      0.99      0.98      1390
            True       0.97      0.93      0.95       480

        accuracy                           0.97      1870
       macro avg       0.97      0.96      0.96      1870
    weighted avg       0.97      0.97      0.97      1870

    Confusion matrix:
      1375  15
      36    444

## Linear SVM

    $ python -m train_model -d ./data.h5 -m -v -c "Linear SVM"
    INFO  Training a Linear SVM classifier with default hyperparameters.
    INFO  Training time: 50.44s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.91      0.98      0.94      1390
            True       0.94      0.71      0.81       480

        accuracy                           0.91      1870
       macro avg       0.92      0.84      0.87      1870
    weighted avg       0.91      0.91      0.91      1870

    Confusion matrix:
      1367  23
      141   339

## Nearest Neighbors

    $ python -m train_model -d ./data.h5 -m -v -c "Nearest Neighbors"
    INFO  Training a Nearest Neighbors classifier with default hyperparameters.
    INFO  Training time: 3.56s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.98      0.99      0.98      1390
            True       0.97      0.93      0.95       480

        accuracy                           0.97      1870
       macro avg       0.97      0.96      0.97      1870
    weighted avg       0.97      0.97      0.97      1870

    Confusion matrix:
      1375  15
      34    446

## Neural Net

    $ python -m train_model -d ./data.h5 -m -v -c "Neural Net"
    INFO  Training a Neural Net classifier with default hyperparameters.
    INFO  Training time: 839.60s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.98      0.99      0.99      1390
            True       0.97      0.95      0.96       480

        accuracy                           0.98      1870
       macro avg       0.97      0.97      0.97      1870
    weighted avg       0.98      0.98      0.98      1870

    Confusion matrix:
      1374  16
      24    456

## Random Forest

    $ python -m train_model -d ./data.h5 -m -v -c "Random Forest"
    INFO  Training a Random Forest classifier with default hyperparameters.
    INFO  Training time: 7.13s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.99      0.99      0.99      1390
            True       0.98      0.97      0.97       480

        accuracy                           0.99      1870
       macro avg       0.99      0.98      0.98      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1382  8
      16    464

## RBF SVM

    $ python -m train_model -d ./data.h5 -m -v -c "RBF SVM"
    INFO  Training a RBF SVM classifier with default hyperparameters.
    INFO  Training time: 48.45s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.99      0.99      0.99      1390
            True       0.98      0.98      0.98       480

        accuracy                           0.99      1870
       macro avg       0.99      0.99      0.99      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1382  8
      11    469

## SGD

    $ python -m train_model -d ./data.h5 -m -v -c "SGD"
    INFO  Training a SGD classifier with default hyperparameters.
    INFO  Training time: 3.69s
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.99      1.00      1.00      1390
            True       0.99      0.99      0.99       480

        accuracy                           0.99      1870
       macro avg       0.99      0.99      0.99      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1385  5
      7     473

### Grid search

    $ python -m train_model -d ./data.h5 -m -v -c SGD -g 50
    INFO  Performing grid search for the preprocessing step with a SGD classifier.
    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    INFO  Training time: 1631.98s
    INFO  Search Accuracy: 0.386381615408105
    INFO  Best parameters:
    {
      "vect__ngram_range": [
        1,
        2
      ],
      "vect__max_features": 50000,
      "vect__max_df": 0.75,
      "tfidf__use_idf": true,
      "tfidf__norm": "l2",
      "process__strip_header": true,
      "process__replace_urls": true,
      "process__replace_numbers": true,
      "process__remove_punctuation": true,
      "process__lowercase": false,
      "clf__penalty": "l2",
      "clf__max_iter": 100,
      "clf__alpha": 1e-06
    }
    INFO  Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.99      1.00      0.99      1390
            True       0.99      0.98      0.99       480

        accuracy                           0.99      1870
       macro avg       0.99      0.99      0.99      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1384  6
      8     472
