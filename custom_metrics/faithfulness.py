"""
Implementation of faithfulness, proposed by https://arxiv.org/abs/1806.07538
Implementation from AIX360: https://github.com/Trusted-AI/AIX360
For each datapoint, compute the correlation between the weights of the 
feature attribution algorithm, and the effect of the features on the 
performance of the model.
TODO: add conditional expectation reference values
"""

import numpy as np


class Faithfulness:
    def __init__(self, model, trained_model, dataset, version="dec", conditional="observational", **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
        self.version = version
        assert version in ['inc', 'dec']
        self.conditional = conditional
        assert conditional in ['observational', 'interventional']

    def evaluate(self, X, y, feature_weights, ground_truth_weights, X_train=None, y_train=None, n_sample=100, X_train_feature_weights=None):
        X = X.values
        num_datapoints, num_features = X.shape
        absolute_weights = abs(feature_weights)

        # compute the base values of each feature
        avg_feature_values = X.mean(axis=0)

        faithfulnesses = []
        for i in range(num_datapoints):
            """
            for each datapoint i, compute the correlation between feature weights
            and the delta in prediction when ablating each feature with replacement
            """
            # original prediction
            y_pred = np.squeeze(self.trained_model.predict(np.array([X[i]])))
            if self.version == 'inc':
                y_pred = np.mean(np.squeeze(self.trained_model.predict(X)))
            # D new predictions (ablate one feature at a time)
            y_preds_new = np.zeros_like(X[i])
            for j in range(num_features):
                # generate a mask
                mask = np.ones_like(X[i])
                mask[j] = 0
                if self.version == 'inc':
                    mask = 1 - mask
                if self.conditional == "observational":
                    # sample n_sample datapoints with feature j ablated
                    x_sampled, _ = self.dataset.generate(mask=mask, x=X[i], n_sample=n_sample)
                    # compute mean over n
                    y_preds_new[j] = np.mean(np.squeeze(self.trained_model.predict(x_sampled)))
                elif self.conditional == "interventional":
                    x_cond = avg_feature_values
                    x_cond[mask.astype(bool)] = X[i][mask.astype(bool)]
                    y_preds_new[j] = self.trained_model.predict([x_cond])[0]
            
            deltas = [abs(y_pred - y_preds_new[j]) for j in range(num_features)]
            faithfulness = np.corrcoef(absolute_weights[i], deltas)[0, 1]
            if np.isnan(faithfulness) or not np.isfinite(faithfulness):
                faithfulness = 0
            faithfulnesses.append(faithfulness)

        return np.mean(faithfulnesses)
