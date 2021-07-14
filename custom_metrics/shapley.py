"""
Compute MSE of feature weights agains ground truth feature weights
Mean is computed over dimension (D) and number of samples (N)
"""

import numpy as np


class Shapley:
    def __init__(self, model, trained_model, dataset, mode="mse", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.mode = mode

    def evaluate(self, X, y, feature_weights, ground_truth_weights, X_train=None, y_train=None, X_train_feature_weights=None):
        if self.mode == "mse":
            error = np.sum(np.square(feature_weights - ground_truth_weights))
        elif self.mode == "rmse":
            error = np.sqrt(np.sum(np.square(feature_weights - ground_truth_weights)))
        elif self.mode == "mae":
            error = np.sum(np.abs(feature_weights - ground_truth_weights))
        # compute mean over D and N
        error = error / (feature_weights.shape[0] * feature_weights.shape[1])

        return error
