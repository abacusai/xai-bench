"""
Implementation of monotonicity, proposed by https://arxiv.org/abs/1905.12698
Implementation from AIX360: https://github.com/Trusted-AI/AIX360
Check whether iteratively adding features from least weighted feature to most
weighted feature, causes the prediction to monotonically improve.
TODO: add other types of reference values

Note: the default version measures the fraction of datapoints that *exactly*
satisfy monotonicity. Setting avg=True is a bit more robust, since it measures
how monotone each datapoint is. Both versions perform poorly on datasets where
multiple features have roughly the same weights.
"""

import numpy as np
import copy
from .roar import split_data

class ROARMonotonicity:
    def __init__(self, model, trained_model, dataset=None, version='inc', conditional="observational", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.version = version
        self.conditional = conditional
        assert conditional in ['observational', 'interventional']

    def evaluate(self, X, y, feature_weights, ground_truth_weights, avg=True, X_train=None, y_train=None, n_sample=100, X_train_feature_weights=None):
        X = X.values
        num_datapoints, num_features = X.shape[0]+X_train.shape[0], X.shape[1]
        absolute_weights = abs(np.concatenate([X_train_feature_weights, feature_weights], axis = 0))

        # compute the base values of each feature
        avg_feature_values = X.mean(axis=0)

        num_tests = len(X)
        y_preds = np.squeeze(self.trained_model.predict(X))
        y_preds_new = np.zeros((num_features+1, num_tests))

        # for the first round mask is zeros
        X_new = np.concatenate([copy.deepcopy(X_train), copy.deepcopy(X)], axis = 0)
        y_new = np.concatenate([copy.deepcopy(y_train), copy.deepcopy(y)], axis = 0)
        for i in range(num_datapoints):
            X_new[i] = self.dataset.generator.computeexpectation(mask=np.zeros_like(X[0]), x=X_new[i])
        
        X_train_new, y_train_new, X_test, y_test = X_new[:len(X_train)], y_new[:len(X_train)], X_new[len(X_train):], y_new[len(X_train):]
        model_new = copy.deepcopy(self.model)
        model_new = model_new.train(X_train_new, y_train_new.ravel())
        y_preds_new[0] = np.squeeze(model_new.predict(X_test))

        for j in range(num_features):
            X_new = np.concatenate([copy.deepcopy(X_train), copy.deepcopy(X)], axis = 0)
            y_new = np.concatenate([copy.deepcopy(y_train), copy.deepcopy(y)], axis = 0)
            for i in range(num_datapoints):
                mask = np.zeros_like(X[0], dtype=np.int32)
                sorted_weight_indices = np.argsort(absolute_weights[i])
                if self.version == 'dec':
                    sorted_weight_indices = sorted_weight_indices[::-1]
                mask[sorted_weight_indices[:j+1]] = 1
                if self.conditional == "observational":
                    X_new[i] = self.dataset.generator.computeexpectation(mask=mask, x=X_new[i])
                elif self.conditional == "interventional":
                    X_new[i][~mask.astype(bool)] = avg_feature_values[~mask.astype(bool)]
            
            X_train_new, y_train_new, X_test, y_test = X_new[:len(X_train)], y_new[:len(X_train)], X_new[len(X_train):], y_new[len(X_train):]
            model_new = copy.deepcopy(self.model)
            model_new = model_new.train(X_train_new, y_train_new.ravel())
            y_preds_new[j+1] = np.squeeze(model_new.predict(X_test))

        deltas = [[abs(y_preds_new[j+1][i]-y_preds_new[j][i]) for j in range(num_features)] for i in range(num_tests)]

        monotonicities = []
        for d in deltas:
            if self.version == 'dec':
                d = d[::-1]
            if avg:
                monotonicity = sum(np.diff(d) >= 0) / (num_features-1)
            else:
                monotonicity = int(np.all(np.diff(d) >= 0))
            monotonicities.append(monotonicity)

        return np.mean(monotonicities)

                