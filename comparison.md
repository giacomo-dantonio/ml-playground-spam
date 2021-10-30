# Comparison of the different classifiers

## AdaBoost

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "AdaBoost"
    Training a AdaBoost classifier with default hyperparameters.
    Training time: 17.00s
    Metrics computed on the test set
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

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "Decision Tree"
    Training a Decision Tree classifier with default hyperparameters.
    Training time: 8.83s
    Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.98      0.99      0.98      1390
            True       0.96      0.94      0.95       480

        accuracy                           0.97      1870
       macro avg       0.97      0.96      0.96      1870
    weighted avg       0.97      0.97      0.97      1870

    Confusion matrix:
      1371  19
      31    449

## Linear SVM

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "Linear SVM"
    Training a Linear SVM classifier with default hyperparameters.
    Training time: 49.54s
    Metrics computed on the test set
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

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "Nearest Neighbors"
    Training a Nearest Neighbors classifier with default hyperparameters.
    Training time: 3.60s
    Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.98      0.99      0.98      1390
            True       0.97      0.93      0.95       480

        accuracy                           0.97      1870
       macro avg       0.97      0.96      0.97      1870
    weighted avg       0.97      0.97      0.97      1870

    Confusion matrix:
      1375  15
      34    446

## Random Forest

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "Random Forest"
    Training a Random Forest classifier with default hyperparameters.
    Training time: 6.73s
    Metrics computed on the test set
                  precision    recall  f1-score   support

           False       0.99      0.99      0.99      1390
            True       0.99      0.97      0.98       480

        accuracy                           0.99      1870
       macro avg       0.99      0.98      0.98      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1383  7
      16    464

## RBF SVM

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "RBF SVM"
    Training a RBF SVM classifier with default hyperparameters.
    Training time: 47.31s
    Metrics computed on the test set
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

    (.venv) giacomo@sisyphos:~/ml/ml-playground-spam$ python -m train_model -d ./data.h5 -m -v -c "SGD"
    Training a SGD classifier with default hyperparameters.
    Training time: 3.75s
    Metrics computed on the test set
                  precision    recall  f1-score   support

           False       1.00      0.99      1.00      1390
            True       0.99      0.99      0.99       480

        accuracy                           0.99      1870
       macro avg       0.99      0.99      0.99      1870
    weighted avg       0.99      0.99      0.99      1870

    Confusion matrix:
      1383  7
      6     474
