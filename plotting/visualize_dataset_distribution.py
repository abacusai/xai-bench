import argparse
import sys
sys.path.append("..")
from src import datasets
import matplotlib.pyplot as plt
fg_w = 5.5
fg_h = 4.5
num_samples = 1000000
mode = "regression"
n_bins = [0.1*x for x in range(-60,60,1)]
rho = 0.0
data_gaussian_kwargs = {
    "mu": "np.zeros(5)",
    # "sigma": "np.identity(4)",
    "dim": 5,
    "rho": rho,
    "weight": "np.array([4, 3, 2, 1, 0])",
    "noise": 0.01,
    "num_train_samples": num_samples,
    "num_val_samples": 100
}

data_mixture_kwargs = {
    "mus": "[1 * np.ones(5), -1 * np.ones(5)]",
    # "sigma": "np.identity(4)",
    "dim": 5,
    "rho": rho,
    "weight": "np.array([4, 3, 2, 1, 0])",
    "noise": 0.01,
    "num_train_samples": num_samples,
    "num_val_samples": 100
}

for k in datasets.valid_datasets[mode]:
    if "gaussian" in k:
        data = datasets.Data(k, mode=mode, **data_gaussian_kwargs)
        y = data.data[1]
        plt.figure(figsize = (fg_w,fg_h))
        plt.hist(y,bins=n_bins,)
        plt.savefig(f'visualize_dataset/{k}_{data_gaussian_kwargs["rho"]}.pdf')
        plt.clf()
    elif "mixture" in k:
        data = datasets.Data(k, mode=mode, **data_mixture_kwargs)
        y = data.data[1]
        plt.figure(figsize = (fg_w,fg_h))
        plt.hist(y,bins=n_bins)
        plt.savefig(f'visualize_dataset/{k}_{data.data_class.rho}.pdf')
        plt.clf()