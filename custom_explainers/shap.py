import shap
from shap import KernelExplainer


class Shap:
    def __init__(self, f, X):
        self.f = f
        self.X = X
        self.explainer = shap.Explainer(self.f, self.X)

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values


class KernelShap:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.explainer = shap.KernelExplainer(self.f, self.X, **kwargs)

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values
