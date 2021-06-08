"""
We generate datasets in two steps:

1. generate D dimentional feature
2. use the feature to generate groundtruth targets

"""

import scipy.special
import numpy as np
import itertools
import copy
import pandas as pd

from .custom_dataset import CustomDataset
from .synthetic_gaussian import MultivariateGaussian
from scipy.stats import multivariate_normal

class MixtureOfGaussian:
    def __init__(self, mus, sigmas, dim, k):
        self.mus = mus
        self.sigmas = sigmas
        self.dim = dim
        self.k = k # k mixtures

        self.gaussians = []

        for i in range(k):
            # print(self.mus[i])
            # print(self.sigmas[i])
            gaussian = MultivariateGaussian(self.mus[i],self.sigmas[i],self.dim)
            self.gaussians.append(gaussian)

    def generateconditional(self, mask, x, n_sample):
        # x is the datapoint
        # mask is a binary mask indicating which dimensions to fix
        X = np.zeros((n_sample,self.dim))
        # return the full distribution
        if len(np.where(mask == 0)[0]) == len(mask):
            sample_k = np.rand.randint(self.k)
            g = self.gaussians[sample_k]
            X += np.random.multivariate_normal(g.mu, g.sigma, n_sample)
        # return a datapoint since everything is fixed
        elif len(np.where(mask > 0)[0]) == len(mask):
            X = np.zeros((n_sample, len(mask)))
            X[:, :] = x
            # print('everything is fixed, p(x=x*) = 1')
        # generate conditional distribution
        else:
            X_cond = np.zeros((n_sample, len(np.where(mask == 0)[0])))
            # find proper mu_1, mu_2, sgima_12_22_inv, and sigma_c
            index = self.gaussians[0].mask_dict[str(mask)]  # find the right cache index
            fixed_indices = np.where(mask)
            variable_indices = np.where(mask == 0)
            a = x[fixed_indices]
            pi_denom = 0
            X_temp_list = []
            pi_list = []
            for i in range(self.k):
                g = self.gaussians[i]
                mu = g.mu_1[index] + np.matmul(
                    g.sigma_12_22_pinv[index], (a - g.mu_2[index])
                )
                p = multivariate_normal(g.mu_2[index],g.sigma_22[index])
                pi = p.pdf(a)
                pi_denom += pi
                pi_list.append(pi)
                X_temp = np.random.multivariate_normal(mu, g.sigma_c[index], n_sample)
                X_temp_list.append(X_temp)

            pi_arr_norm = np.array(pi_list) / pi_denom
            sample_k = np.argmax(np.random.multinomial(1, pvals=pi_arr_norm))
            X_cond += X_temp_list[sample_k]

            X[:, list(variable_indices[0])] = X_cond
            X[:, list(fixed_indices[0])] = x[
                fixed_indices
            ]
        return X

    def computeexpectation(self, mask, x):
        # computes conditional expectation given mask and x
        X = np.zeros_like(x)
        if len(np.where(mask == 0)[0]) == len(mask):
            for i in range(self.k):
                g = self.gaussians[i]
                X += np.array(g.mu)
            X /= self.k
            # print('return expectation')
        # return a datapoint since everything is fixed
        elif len(np.where(mask > 0)[0]) == len(mask):
            X = x
            # print('everything is fixed, p(x=x*) = 1')
        # generate conditional mean
        else:
            # find proper mu_1, mu_2, sgima_12_22_inv, and sigma_c
            Mu_cond = np.zeros((1, len(np.where(mask == 0)[0])))
            index = self.gaussians[0].mask_dict[str(mask)]  # find the right cache index
            fixed_indices = np.where(mask)
            variable_indices = np.where(mask == 0)
            a = x[fixed_indices]
            pi_denom = 0
            for i in range(self.k):
                g = self.gaussians[i]
                mu = g.mu_1[index] + np.matmul(
                    g.sigma_12_22_pinv[index], (a - g.mu_2[index])
                )
                p = multivariate_normal(g.mu_2[index],g.sigma_22[index])
                pi = p.pdf(a)
                pi_denom += pi
                Mu_cond += pi * mu
            Mu_cond /= pi_denom
            X[list(variable_indices[0])] = Mu_cond
            X[list(fixed_indices[0])] = x[fixed_indices]

        return X


class GMLinearRegression(CustomDataset):
    def __init__(
        self,
        mus,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigmas=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples)
        self.mus = mus
        self.dim = dim
        self.K = len(mus)
        self.sigmas = [np.identity(self.dim) for i in range(len(mus))] if sigmas is None else sigmas
        self.rho = rho
        if rho:
            self.sigmas = [sigma + (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho for sigma in self.sigmas]
        if len(weight.shape) == 1:
            weight = np.expand_dims(weight, 1)
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MixtureOfGaussian(
            mus=self.mus, sigmas=self.sigmas, dim=self.dim, k=self.K
        )

    def getdim(self):
        return self.dim

    def getweight(self):
        return self.weight

    def generatetarget(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        Y = np.matmul(X, self.weight) + np.random.normal(
            scale=self.noise, size=(X.shape[0], 1)
        )
        Y -= np.mean(Y)
        Y /= np.std(Y)
        return Y

    def generate(self, mask=None, x=None, n_sample=1):
        # if nothing is passed, it will generate a single data point from the original gaussian
        if mask is None:
            mask = self.default_mask
        else:
            mask = mask.astype(np.int)

        if x is None:
            x = self.default_mask

        X = self.generator.generateconditional(mask=mask, x=x, n_sample=n_sample)
        Y = self.generatetarget(X)

        return X, Y


class GMPiecewiseConstantRegression(CustomDataset):
    def __init__(
        self,
        mus,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigmas=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples)
        self.mus = mus
        self.dim = dim
        self.K = len(mus)
        self.sigmas = [np.identity(self.dim) for i in range(len(mus))] if sigmas is None else sigmas
        self.rho = rho
        if rho:
            self.sigmas = [sigma + (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho for sigma in self.sigmas]
        if len(weight.shape) == 1:
            weight = np.expand_dims(weight, 1)
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MixtureOfGaussian(
            mus=self.mus, sigmas=self.sigmas, dim=self.dim, k=self.K
        )
        self.num_piece = 3  # number of piecewise constant functions

    def getdim(self):
        return self.dim

    def getweight(self):
        return self.weight

    def generatetarget(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        Y = np.zeros((X.shape[0], 1))
        # print(X.shape)

        for i in range(min(self.dim, self.num_piece)):

            x = X[:, i]
            if i == 0:
                p = np.piecewise(
                    x, [x < 0, x >= 0], [-1, 1]
                )  # can be replaced by sign()
            elif i == 1:
                p = np.piecewise(
                    x,
                    [x < -0.5, (x >= 0.5) * (x < 0), (x >= 0) * (x < 0.5), x >= 0.5],
                    [-2, -1, 1, 2],
                )
            elif i == 2:
                p = (2 * np.cos(x * np.pi)).astype(np.int)
                p[np.where(p == 0)] = 1  # with small probability cos(x) == 0

            p = np.expand_dims(p, axis=1)
            # print('piecewise function: {}'.format(i))
            # print(Y.shape)
            # print(p)
            Y = Y + p

        Y = Y + np.random.normal(scale=self.noise, size=(X.shape[0], 1))
        Y -= np.mean(Y)
        Y /= np.std(Y)
        return Y

    def generate(self, mask=None, x=None, n_sample=1):
        # if nothing is passed, it will generate a single data point from the original gaussian
        if mask is None:
            mask = self.default_mask
        else:
            mask = mask.astype(np.int)
        if x is None:
            x = self.default_mask

        X = self.generator.generateconditional(mask=mask, x=x, n_sample=n_sample)
        Y = self.generatetarget(X)

        # print("X:\n {} \n Y:\n {}".format(X,Y))
        return X, Y


class GMNonlinearAdditiveRegression(CustomDataset):
    """
    This class is a generalized version of the "Nonlinear Additive" dataset from the paper:
    "Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"
    The original dataset generates a 10 dimentional independent gaussian feature vector.
    Here we allow the features to be correlated.
    """

    def __init__(
        self,
        mus,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigmas=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples)
        self.mus = mus
        self.dim = dim
        self.K = len(mus)
        self.sigmas = [np.identity(self.dim) for i in range(len(mus))] if sigmas is None else sigmas
        self.rho = rho
        if rho:
            self.sigmas = [sigma + (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho for sigma in self.sigmas]
        if len(weight.shape) == 1:
            weight = np.expand_dims(weight, 1)
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MixtureOfGaussian(
            mus=self.mus, sigmas=self.sigmas, dim=self.dim, k=self.K
        )
        self.num_true_feature = 4  # 4 true components are used

    def getdim(self):
        return self.dim

    def getweight(self):
        return self.weight

    def generatetarget(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.values
        Y = np.zeros((X.shape[0], 1))

        for i in range(min(self.dim, self.num_true_feature)):

            x = X[:, i]
            if i == 0:
                p = np.sin(1.0 * x)
            elif i == 1:
                p = 1 * np.abs(x)
            elif i == 2:
                p = x**2
            elif i == 3:
                p = np.exp(-x)

            p = np.expand_dims(p, axis=1)
            Y = Y + p
        Y = Y + np.random.normal(scale=self.noise, size=(X.shape[0], 1))
        Y -= np.mean(Y)
        Y /= np.std(Y)
        return Y

    def generate(self, mask=None, x=None, n_sample=1):
        # if nothing is passed, it will generate a single data point from the original gaussian
        if mask is None:
            mask = self.default_mask
        else:
            mask = mask.astype(np.int)

        if x is None:
            x = self.default_mask

        X = self.generator.generateconditional(mask=mask, x=x, n_sample=n_sample)
        Y = self.generatetarget(X)
        return X, Y

# if __name__ == "__main__":
#     # test
#     N = 10
#     D = 5
#     rho = 0.0
#     mean = np.ones(D)
#     weight = np.array([D-1-i for i in range(D)]).astype(float)
#     cov = np.squeeze( (np.ones((D,D)) - np.identity(D)) * rho + np.identity(D) )
#     means = [10 * mean, -15 * mean]
#     covs= [cov,cov]
#     print(cov.shape)
#     a = MixtureOfGaussian(means,covs,D,2)
#     X = np.array([0.1,0.2,0.3,100.0,12.0])
#     Mask = np.array([1,0,1,0,1])
#     # a = MultivariateGaussian(mu=mean,sigma=cov,dim=5)

#     x = a.generateconditional(mask=Mask,
#                             x=X,
#                             n_sample=1000)
#     y = a.computeexpectation(mask=Mask,
#                                     x=X)
#     m = np.mean(x,axis=0)
#     # sd = np.std(x,axis=0)
#     print(y)
#     print(m)
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(2,1)
#     x = a.generateconditional(mask=np.zeros_like(Mask),x=np.zeros_like(Mask),n_sample=10000)
#     print('std:', np.std(x[:,0]))
#     fig, axs = plt.subplots(1,2)
#     n_bins_y = [x for x in range(0,10,1)]
#     n_bins_y = 40
#     axs[0].hist(x[:,0],bins=n_bins_y)
#     axs[1].hist(x[:,1],bins=n_bins_y)
#     plt.show()
