"""
Compute MSE of feature weights agains ground truth feature weights
Mean is computed over dimension (D) and number of samples (N)
"""

import numpy as np


class ShapleyCorr:
    def __init__(self, model, trained_model, dataset, mode="mse", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.mode = mode

    def evaluate(self, X, y, feature_weights, ground_truth_weights, X_train=None, y_train=None, X_train_feature_weights=None):
        corr = [np.corrcoef(a, b)[0, 1] for a, b in zip(feature_weights, ground_truth_weights)]
        corr = np.nan_to_num(corr, nan=0, posinf=0, neginf=0)
        return np.mean(corr)
