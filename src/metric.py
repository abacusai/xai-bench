from custom_metrics import *

valid_metrics = {
    "faithfulness": Faithfulness,
    "roar_faithfulness": ROARFaithfulness,
    "roar_monotonicity": ROARMonotonicity,
    "monotonicity": Monotonicity,
    "roar": Roar,
    "shapley": Shapley,
    "shapley_corr": ShapleyCorr,
    "infidelity": Infidelity
}


class Metric:
    def __init__(self, name, **kwargs):
        if name not in valid_metrics.keys():
            raise NotImplementedError("This metric is not supported at the moment.")
        self.name = name
        self.metric = lambda model, trained_model, data_class: valid_metrics[name](model, trained_model, data_class, **kwargs)
        self.evaluate = valid_metrics[name].evaluate
