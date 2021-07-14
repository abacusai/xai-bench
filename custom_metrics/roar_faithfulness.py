"""
Implementation of faithfulness, proposed by https://arxiv.org/abs/1806.07538
Implementation from AIX360: https://github.com/Trusted-AI/AIX360
For each datapoint, compute the correlation between the weights of the 
feature attribution algorithm, and the effect of the features on the 
performance of the model.
TODO: add conditional expectation reference values
"""

import numpy as np
import copy
from .roar import split_data

class ROARFaithfulness:
    def __init__(self, model, trained_model, dataset, version="dec", conditional="observational", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.version = version
        assert version in ['inc', 'dec']
        self.conditional = conditional
        assert conditional in ['observational', 'interventional']

    def evaluate(self, X, y, feature_weights,  ground_truth_weights, X_train=None, y_train=None, n_sample=100, X_train_feature_weights=None):
        X = X.values
        X_train = X_train.values
        num_datapoints, num_features = X.shape[0]+X_train.shape[0], X.shape[1]
        absolute_weights = abs(np.concatenate([X_train_feature_weights, feature_weights], axis = 0))

        # compute the base values of each feature
        avg_feature_values = X.mean(axis=0)

        num_tests = len(X)
        y_preds = np.squeeze(self.trained_model.predict(X))
        if self.version == 'inc':
            y_preds = np.full((num_tests, np.squeeze(y[0].shape)), np.mean(np.squeeze(self.trained_model.predict(X))))
        y_preds_new = np.zeros((num_features, num_tests))

        for j in range(num_features):
            X_new = np.concatenate([copy.deepcopy(X_train), copy.deepcopy(X)], axis = 0)
            y_new = np.concatenate([copy.deepcopy(y_train), copy.deepcopy(y)], axis = 0)
            for i in range(num_datapoints):
                sorted_indices = np.argsort(absolute_weights[i])[::-1]
                indices_to_remove = sorted_indices[j]
                # remove the features and replace with average
                mask = np.ones_like(X_new[i], dtype=np.int32)
                mask[indices_to_remove] = 0
                if self.version == 'inc':
                    mask = 1 - mask
                if self.conditional == "observational":
                    X_new[i] = self.dataset.generator.computeexpectation(mask=mask, x=X_new[i])
                elif self.conditional == "interventional":
                    X_new[i][~mask.astype(bool)] = avg_feature_values[~mask.astype(bool)]
            
            X_train_new, y_train_new, X_test, y_test = X_new[:len(X_train)], y_new[:len(X_train)], X_new[len(X_train):], y_new[len(X_train):]
            model_new = copy.deepcopy(self.model)
            model_new = model_new.train(X_train_new, y_train_new.ravel())
            y_preds_new[j] = np.squeeze(model_new.predict(X_test))
        
        deltas = [[abs(y_preds[i] - y_preds_new[j][i]) for j in range(num_features)] for i in range(num_tests)]

        faithfulness = [np.corrcoef(absolute_weights[i], np.squeeze(deltas[i]))[0, 1] for i in range(num_tests)]
        faithfulness = np.nan_to_num(faithfulness, nan=0, posinf=0, neginf=0)
        return np.mean(faithfulness)