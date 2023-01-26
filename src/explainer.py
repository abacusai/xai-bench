from custom_explainers import *

valid_explainers = {
    "shap_trustyai": shap.KernelShapTrustyAI,
    "shap": KernelShap,
    "random": Random,
    "lime": Lime,
    "lime_trustyai": LimeTrustyAI,
}


class Explainer:
    def __init__(self, name, **kwargs):
        if name not in valid_explainers.keys():
            raise NotImplementedError(
                f"This explainer ({name}) is not supported at the moment. Explainers supported are {list(valid_explainers.keys())}"
            )
        self.name = name
        def explainerinit(clf, data, modelname):
            kwargs['model_name'] = modelname
            return valid_explainers[name](clf, data, **kwargs)

        self.explainer = explainerinit
        self.label = kwargs.get("label", name)
