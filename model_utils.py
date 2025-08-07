# model_utils.py

import numpy as np

class ThresholdedClassifier:
    """
    A wrapper for classifiers that allows setting a custom decision threshold
    for binary classification.
    """
    def __init__(self, model, threshold=0.35):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        """
        Predict binary labels using the custom threshold.
        """
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Return the probability estimates from the base model.
        """
        return self.model.predict_proba(X)

    def fit(self, X, y):
        """
        Fit the base model.
        """
        self.model.fit(X, y)
        return self

    def score(self, X, y):
        """
        Return the accuracy score of the base model.
        """
        return self.model.score(X, y)
