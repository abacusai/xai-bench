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


class Monotonicity:
    def __init__(self, model, trained_model, dataset=None, version='inc', conditional="observational", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.version = version
        self.conditional = conditional
        assert conditional in ['observational', 'interventional']

    def evaluate(self, X, y, feature_weights, ground_truth_weights, avg=True, X_train=None, y_train=None, n_sample=100, X_train_feature_weights=None):
        X = X.values
        num_datapoints, num_features = X.shape
        absolute_weights = abs(feature_weights)

        # compute the base values of each feature
        avg_feature_values = X.mean(axis=0)

        monotonicities = []

        y_preds_mean = np.mean(np.squeeze(self.trained_model.predict(X)))
        for i in range(num_datapoints):
            mask = np.zeros_like(X[i])
            sorted_weight_indices = np.argsort(absolute_weights[i])
            if self.version == 'dec':
                sorted_weight_indices = sorted_weight_indices[::-1]
            y_preds_new = np.zeros(len(X[i])+1)
            y_preds_new[0] = y_preds_mean

            for j in sorted_weight_indices:
                mask[j] = 1
                if self.conditional == "observational":
                    x_sampled, _ = self.dataset.generate(mask=mask, x=X[i], n_sample=n_sample)
                    y_preds_new[j+1] = np.mean(np.squeeze(self.trained_model.predict(x_sampled)))
                elif self.conditional == "interventional":
                    x_cond = avg_feature_values
                    x_cond[mask.astype(bool)] = X[i][mask.astype(bool)]
                    y_preds_new[j+1] = self.trained_model.predict([x_cond])[0]

            deltas = np.abs(np.diff(y_preds_new))
            if self.version == 'dec':
                deltas = deltas[::-1]
            if avg:
                monotonicity = sum(np.diff(deltas) >= 0) / (num_features-1)
            else:
                monotonicity = int(np.all(np.diff(deltas) >= 0))

            monotonicities.append(monotonicity)
        return np.mean(monotonicities)
