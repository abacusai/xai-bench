import numpy as np


class SHAP_Random:
    """Simply returns random (normally distributed) feature attributions.

    This is only for benchmark comparisons. It supports both fully random attributions and random
    attributions that are constant across all explainations.
    """

    def __init__(self, constant=False):
        self.constant = constant
        self.constant_attributions = None

    def attributions(self, X):
        if self.constant:
            if self.constant_attributions is None:
                self.constant_attributions = np.random.randn(X.shape[1])
            return np.tile(self.constant_attributions, (X.shape[0], 1))
        else:
            return np.random.randn(*X.shape)


class Random:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.explainer = SHAP_Random().attributions

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values = np.zeros(
            x.shape[0]
        )  # TODO: maybe we might want to change this later
        shap_values = shap_values
        return shap_values
