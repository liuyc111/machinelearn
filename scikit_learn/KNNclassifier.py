import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k) -> None:
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._Y_train = None

    def fit(self, X_train, Y_train):
        assert X_train.shape[0] == Y_train.shape[0], "the size of X_train must be equal to the size of Y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be least k"
        self._X_train = X_train
        self._Y_train = Y_train

    def predict(self, x_predict):
        return [self._predict(x) for x in x_predict]

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x_train-x)**2))
                     for x_train in self._X_train]
        return Counter([self._Y_train[item] for item in np.argsort(distances)[:self.k]]).most_common(1)[0][0]
