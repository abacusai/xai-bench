"""
Implementation of ROAR, remove and retrain - https://arxiv.org/abs/1806.10758
Remove the features deemed most important, then remove them from the data and retrain.
A higher roar score is better.
"""

import numpy as np
import copy

from numpy.core.numeric import indices

# this is the set of cutoffs used in the ROAR paper
CUTOFFS = [0, 0.1, 0.3, 0.5, 0.7, 0.9]


def split_data(X, y, train_test_split=0.8):
    # split X and y into train/test data
    split = int(train_test_split * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, y_train, X_test, y_test


def evaluate_model(y_test, test_pred):
    # compute the accuracy based on predictions and ground truth
    return np.mean([abs(y_test[i] - test_pred[i]) for i in range(len(y_test))])


def auc(x, y):
    # compute the area under the curve of a scatterplot
    area = 0
    for i in range(1, len(x)):
        length = (y[i] + y[i - 1]) / 2
        width = x[i] - x[i - 1]
        area += length * width
    return area


class Roar:
    def __init__(self, model, trained_model, dataset=None, conditional="observational", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.conditional = conditional
        assert conditional in ['observational', 'interventional']
    
    def evaluate(self, X, y, feature_weights, ground_truth_weights, avg=True, X_train=None, y_train=None, n_sample=100, X_train_feature_weights=None):
    # def evaluate(self, X, y, feature_weights, ground_truth_weights, X_train, y_train):
        X = X.values
        
        num_datapoints, num_features = X.shape[0]+X_train.shape[0], X.shape[1]
        absolute_weights = abs(np.concatenate([X_train_feature_weights, feature_weights], axis = 0))

        # compute the base values of each feature
        avg_feature_values = X.mean(axis=0)
        # avg_feature_values = np.ones_like(avg_feature_values) * 1
        roar_values = []
        losses = []
        for cutoff_percent in CUTOFFS:
            cutoff = int(cutoff_percent * num_features)
            X_new = np.concatenate([copy.deepcopy(X_train), copy.deepcopy(X)], axis = 0)
            y_new = np.concatenate([copy.deepcopy(y_train), copy.deepcopy(y)], axis = 0)

            for i in range(num_datapoints):
                """
                For each datapoint i, iteratively remove features
                of decreasing importance, and record the change in performance
                """
                sorted_indices = np.argsort(absolute_weights[i])[::-1]
                indices_to_remove = sorted_indices[:cutoff]
                # remove the features and replace with average
                mask = np.ones_like(X_new[i])
                mask[indices_to_remove] = 0
                if self.conditional == "observational":
                    X_new[i], _ = self.dataset.generate(mask=mask, x=X_new[i], n_sample=1)
                elif self.conditional == "interventional":
                    X_new[i][~mask.astype(bool)] = avg_feature_values[~mask.astype(bool)]

            # train a new model and predict on the test set
            X_train_new, y_train_new, X_test, y_test = X_new[:len(X_train)], y_new[:len(X_train)], X_new[len(X_train):], y_new[len(X_train):]
            model_new = copy.deepcopy(self.model)
            model_new = model_new.train(X_train_new, y_train_new.ravel())
            preds = model_new.predict(X_test)

            loss = evaluate_model(y_test, preds)
            losses.append(loss)

        roar_values.append(auc(CUTOFFS, losses))

        return np.mean(roar_values)
