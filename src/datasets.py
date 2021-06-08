# Originally from SHAP official repo: https://github.com/slundberg/shap

import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.datasets import fetch_openml
import os
import synthetic_datasets
from numpy import array

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# @Yang
valid_datasets = {
    "regression": {
        "gaussianLinear": synthetic_datasets.GaussianLinearRegression,
        "gaussianNonLinearAdditive": synthetic_datasets.GaussianNonlinearAdditiveRegression,
        "gaussianPiecewiseConstant": synthetic_datasets.GaussianPiecewiseConstantRegression,
        "mixtureLinear": synthetic_datasets.GMLinearRegression,
        "mixtureNonLinearAdditive": synthetic_datasets.GMNonlinearAdditiveRegression,
        "mixturePiecewiseConstant": synthetic_datasets.GMPiecewiseConstantRegression
    },
    "classification": {
        "gaussianLinear": synthetic_datasets.GaussianLinearBinary,
        "gaussianNonLinearAdditive": synthetic_datasets.GaussianNonlinearAdditiveBinary,
        "gaussianPiecewiseConstant": synthetic_datasets.GaussianPiecewiseConstantBinary,
    },
}


class Data:
    def __init__(self, name: str, mode: str, **kwargs):
        if name not in valid_datasets[mode].keys():
            raise NotImplementedError(
                f"This dataset is not supported at the moment. Datasets supported are: {list(valid_datasets.keys())}"
            )
        self.name = name
        self.kwargs = kwargs
        data_kwargs = {k: eval(str(v)) for k, v in kwargs.items()}
        self.mode = mode
        self.data = valid_datasets[mode][name](**data_kwargs)
        self.data_class = None  # the underlying data generating class is unknown
        if isinstance(self.data, synthetic_datasets.CustomDataset):
            self.data_class = (
                self.data
            )  # known underlying data generating class used by ground_truth_shap
            self.data = self.data.get_dataset()
            self.val_data = self.data_class.get_dataset(self.data_class.num_val_samples)
            self.data[0].columns = [f"feat_{i}" for i in range(self.data[0].shape[1])]
            self.val_data[0].columns = [
                f"feat_{i}" for i in range(self.val_data[0].shape[1])
            ]
