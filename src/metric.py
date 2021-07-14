import custom_metrics

valid_metrics = {
    "faithfulness": custom_metrics.Faithfulness,
    "roar_faithfulness": custom_metrics.ROARFaithfulness,
    "roar_monotonicity": custom_metrics.ROARMonotonicity,
    "monotonicity": custom_metrics.Monotonicity,
    "roar": custom_metrics.Roar,
    "shapley": custom_metrics.Shapley,
    "shapley_corr": custom_metrics.ShapleyCorr,
    "infidelity": custom_metrics.Infidelity
}


class Metric:
    def __init__(self, name, **kwargs):
        if name not in valid_metrics.keys():
            raise NotImplementedError("This metric is not supported at the moment.")
        self.name = name
        self.metric = lambda model, trained_model, data_class: valid_metrics[name](model, trained_model, data_class, **kwargs)
        self.evaluate = valid_metrics[name].evaluate
