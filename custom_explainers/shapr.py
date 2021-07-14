# Implementation of SHAPR using empirical condtional distribution - https://arxiv.org/pdf/1903.10464.pdf
# method has two parameters K and sigma which are passed as defaults (100, 0.1) to the functions

import scipy.special
import numpy as np
import itertools
from tqdm import tqdm


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000  # approximation of inf with some large weight
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


def get_weights(X_data, s, x, sigma=0.4):
    sample_cov_s = np.linalg.pinv(np.cov(X_data[:, s], rowvar=False))
    D_s = np.einsum(
        "ji,ji->j", np.dot(x[s] - X_data[:, s], sample_cov_s), x[s] - X_data[:, s]
    )
    D_s = np.sqrt(D_s / len(s))
    w_s = np.exp(-np.square(D_s) / (2 * (sigma ** 2)))
    return w_s


def get_weighted_mean(w_s, s, f, x, reference):
    w_s_sort_idx = np.argsort(w_s)[::-1]
    wp_sum, w_sum = 0.0, 0.0
    for idx in w_s_sort_idx:
        x_eval = reference[idx].copy()
        x_eval[s] = x[s]
        wp_sum += w_s[idx] * f(x_eval.reshape(1, -1))
        w_sum += w_s[idx]
    return wp_sum / w_sum


def kernel_shapr(f, x, reference, M, sigma):
    X = np.zeros((2 ** M, M + 1))
    X[:, -1] = 1
    weights = np.zeros(2 ** M)
    y = np.zeros(2 ** M)

    for i, s in enumerate(powerset(range(M))):
        s = list(s)
        w_s = np.ones(len(reference))
        if len(s) > 1:
            w_s = get_weights(reference, s, x, sigma)
        X[i, s] = 1
        weights[i] = shapley_kernel(M, len(s))
        y[i] = get_weighted_mean(w_s, s, f, x, reference)

    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))


class ShapR:
    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X.values
        self.M = X.shape[1]
        self.sigma = 0.4 if 'sigma' not in kwargs else kwargs['sigma']

    def explain(self, x):
        phi = np.zeros((x.shape[0], self.M + 1))
        for idx, x in tqdm(enumerate(x.values)):
            phi[idx] = kernel_shapr(self.f, x, self.X, self.M, self.sigma)
        self.expected_values = phi[:, -1]
        shap_values = phi[:, :-1]
        return shap_values
