from custom_explainers.breakdown import BreakDown
import shap
import custom_explainers

valid_explainers = {
    "shap": custom_explainers.Shap,
    "shapr": custom_explainers.ShapR,
    "kernelshap": custom_explainers.KernelShap,
    "brutekernelshap": custom_explainers.BruteForceKernelShap,
    "random": custom_explainers.Random,
    "lime": custom_explainers.Lime,
    "maple": custom_explainers.Maple,
    "l2x": custom_explainers.L2X,
    "breakdown": custom_explainers.BreakDown,
}


class Explainer:
    def __init__(self, name, **kwargs):
        if name not in valid_explainers.keys():
            raise NotImplementedError(
                f"This explainer is not supported at the moment. Explainers supported are {list(valid_explainers.keys())}"
            )
        self.name = name
        self.explainer = lambda clf, data: valid_explainers[name](clf, data, **kwargs)
