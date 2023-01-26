import shap
import numpy as np
from trustyai.explainers import SHAPExplainer
from trustyai.model import Model

import gc
NSAMPLES = 512

class KernelShap:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.kwargs = kwargs
        self.explainer = shap.KernelExplainer(
            self.f,
            self.X[:kwargs.get("background_size", 100)],
            **kwargs
        )

    def explain(self, x):
        shap_values = self.explainer.shap_values(x, nsamples=self.kwargs.get("samples", "auto"))
        self.expected_values, shap_values = self.explainer.expected_value, shap_values
        return shap_values



class KernelShapTrustyAI:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.model = Model(f, dataframe_input=True)
        self.explainer = SHAPExplainer(
            background=X[:kwargs.get("background_size", 100)],
            **kwargs
        )

    def explain(self, x):
        results = []
        predictions = self.model(x)
        for i in range(len(x)):
            saliency = self.explainer.explain(
                    inputs=x.iloc[i:i + 1],
                    outputs=predictions[i:i + 1],
                    model=self.model).saliency_map()

            output_name = list(saliency.keys())[0]
            results.append(
                [float(pfi.getScore()) for pfi in saliency[output_name].getPerFeatureImportance()[:-1]]
            )
        return np.array(results)
