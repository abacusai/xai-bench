"""
Computing an approximation of mutual information (MI) using the approximation given by
L2X (https://arxiv.org/pdf/1802.07814.pdf).

*NOTE: code in progress*

TODOs:
- once finished, add this metric to metric.py
- determine inputs to evaluate: (X, y) vs feature_weights vs ground_truth_weights
- determine L2X initialization: (X, f) vs (X, y)

"""

import numpy as np

from ..custom_explainers.l2x import L2X


class MI:
    def __init__(self, model, trained_model, k_list=[10]):
        self.model = model
        self.trained_model = trained_model
        self.k_list = k_list

    def set_mi_list(self, X, y):
        """
        Set self.mi_list, where each element contains the top-k features (based on MI),
        for a k in self.k_list.
        """
        # Initialize L2X with X, y

        self.mi_list = []

        # for each k in self.k_list:
        # get the top_k mi features using L2X, append this list to self.mi_list

    def set_explain_list(self, X, y, feature_weights):
        """
        Set self.explain_list, where each element contains the top-k features chosen by
        the explainer, for a k in self.k_list.
        """
        # Get the feature attributions for the explainer

        self.explain_list = []

        # for each k in self.k_list:
        # get the top_k features based on the feature attribution, append this list to
        # self.explain_list

    def get_error_single_k(self, mi_list_k, explain_list_k, mode="jaccard"):
        """
        Return the error for the top k features given by an explainability method.
        """
        # Compute error between two lists based on (e.g.) jaccard distance

        return error

    def get_error_explainer(self):
        """
        Return the error for a single explainer, 
        """
        error_list = []
        for mi_list_k, explain_list_k in zip(self.mi_list, self.explain_list):
            error = self.get_error_single_k(mi_list_k, explain_list_k)
            error_list.append(self.get_error_)

    def evaluate(self, X, y, feature_weights, ground_truth_weights, X_train=None, y_train=None):
        """Return MI error metric for the given explainer feature attribution."""
        self.set_mi_list(X, y)
        self.set_explain_list(X, y, feature_weights)

        error = self.get_error_explainer()

        return error
