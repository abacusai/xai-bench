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
        self.explainer = shap.KernelExplainer(self.f, self.X[:100], **kwargs)

    def explain(self, x):
        shap_values = self.explainer.shap_values(x, nsamples=NSAMPLES)
        self.expected_values, shap_values = self.explainer.expected_value, shap_values
        print(shap_values.shape)
        return shap_values



class KernelShapTrustyAI:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.model = Model(f, dataframe_input=True, arrow=True)
        self.explainer = SHAPExplainer(background=X[:100], batch_size=20, samples=NSAMPLES)

    def explain(self, x):
        results = []
        predictions = self.model(x)
        for i in range(len(x)):
            saliency = self.explainer.explain(
                    inputs=x.iloc[i:i + 1],
                    outputs=predictions[i:i + 1],
                    model=self.model).get_saliencies()
            print("shap explanation",i)
            print(globals())
            output_name = list(saliency.keys())[0]
            results.append(
                [float(pfi.getScore()) for pfi in saliency[output_name].getPerFeatureImportance()[:-1]]
            )
        return np.array(results)
