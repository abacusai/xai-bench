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

from scipy.stats import multivariate_normal
from .custom_dataset import CustomDataset


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def computemusigma(
    mu=None,
    sigma=None,
    mask=None,  # binary mask of which indices to fix
):

    fixed_indices = np.where(mask)
    variable_indices = np.where(1 - mask)

    mu_1 = mu[variable_indices]
    mu_2 = mu[fixed_indices]

    sigma_11 = np.delete(copy.copy(sigma), fixed_indices, 0)
    sigma_11 = np.delete(sigma_11, fixed_indices, 1)

    sigma_22 = np.delete(copy.copy(sigma), variable_indices, 0)
    sigma_22 = np.delete(sigma_22, variable_indices, 1)

    sigma_12 = np.delete(copy.copy(sigma), fixed_indices, 0)
    sigma_12 = np.delete(sigma_12, variable_indices, 1)

    sigma_21 = np.delete(copy.copy(sigma), variable_indices, 0)
    sigma_21 = np.delete(sigma_21, fixed_indices, 1)

    sigma_22_pinv = np.linalg.pinv(sigma_22)

    sigma_12_22_pinv = np.matmul(sigma_12, sigma_22_pinv)
    sigma_12_22_pinv_21 = np.matmul(sigma_12_22_pinv, sigma_21)

    # mu = mu_1 + np.matmul(np.matmul(sigma_12, sigma_22_pinv),(a - mu_2))

    sigma = sigma_11 - sigma_12_22_pinv_21

    # only need mu_1, mu_2, sigma_12_22_pinv, and sigma, returning everything just in case
    return {
        "mu_1": mu_1,
        "mu_2": mu_2,
        "sigma_11": sigma_11,
        "sigma_22": sigma_22,
        "sigma_12": sigma_12,
        "sigma_21": sigma_21,
        "sigma_22_pinv": sigma_22_pinv,
        "sigma_12_22_pinv": sigma_12_22_pinv,
        "sigma_12_22_pinv_21": sigma_12_22_pinv_21,
        "sigma": sigma,
    }


class MultivariateGaussian:
    def __init__(self, mu, sigma, dim, precompute=False):
        self.mu = mu
        self.sigma = sigma
        self.dim = dim
        self.precompute = precompute
        # self.n_sample = n_sample

        def generatemask(self):
            self.mask = np.zeros((2 ** self.dim, self.dim))
            self.mask_dict = {}
            for i, s in enumerate(powerset(range(dim))):
                s = list(s)
                self.mask[i, s] = 1
                h = str(self.mask[i, :].astype(np.int))
                self.mask_dict[h] = i
            # print(self.mask)
        if self.precompute:
            generatemask(self)

            self.mu_1 = []
            self.mu_2 = []

            self.sigma_22 = []
            self.sigma_12_22_pinv = []  # intermediate variable
            self.sigma_c = []  # conditional sigmas

            for m in self.mask:
                results = computemusigma(self.mu, self.sigma, m)
                self.mu_1.append(results["mu_1"])
                self.mu_2.append(results["mu_2"])
                self.sigma_12_22_pinv.append(results["sigma_12_22_pinv"])
                self.sigma_c.append(results["sigma"])
                self.sigma_22.append(results["sigma_22"])
            print("finished initialization")

    def generateconditional(self, mask, x, n_sample):
        # x is the datapoint
        # mask is a binary mask indicating which dimensions to fix

        # return the full distribution
        if len(np.where(mask == 0)[0]) == len(mask):
            X = np.random.multivariate_normal(self.mu, self.sigma, n_sample)
            # print('full distribution',self.mu,self.sigma)
        # return a datapoint since everything is fixed
        elif len(np.where(mask > 0)[0]) == len(mask):
            X = np.zeros((n_sample, len(mask)))
            X[:, :] = x
            # print('everything is fixed, p(x=x*) = 1')
        # generate conditional distribution
        else:
            if self.precompute:
                # find proper mu_1, mu_2, sgima_12_22_inv, and sigma_c
                index = self.mask_dict[str(mask)]  # find the right cache index
                fixed_indices = np.where(mask)
                variable_indices = np.where(mask == 0)
                a = x[fixed_indices]
                mu = self.mu_1[index] + np.matmul(
                    self.sigma_12_22_pinv[index], (a - self.mu_2[index])
                )
                X_cond = np.random.multivariate_normal(mu, self.sigma_c[index], n_sample)
            else:
                fixed_indices = np.where(mask)
                variable_indices = np.where(mask == 0)
                a = x[fixed_indices]
                results = computemusigma(self.mu, self.sigma, mask)
                mu = results['mu_1'] + np.matmul(
                    results["sigma_12_22_pinv"], (a - results["mu_2"])
                )
                X_cond = np.random.multivariate_normal(mu, results["sigma"], n_sample)
            X = np.zeros((n_sample, len(mask)))
            # print(list(variable_indices[0]))
            # print(X[:,variable_indices].shape)
            X[:, list(variable_indices[0])] = X_cond
            X[:, list(fixed_indices[0])] = x[
                fixed_indices
            ]  # np.tile(x[fixed_indices],(n_sample,1))
            # print('generating conditional distribution')
        return X

    def computeexpectation(self, mask, x):
        # computes conditional expectation given mask and x
        if len(np.where(mask == 0)[0]) == len(mask):
            X = np.array(self.mu)
            # print('return expectation')
        # return a datapoint since everything is fixed
        elif len(np.where(mask > 0)[0]) == len(mask):
            X = x
            # print('everything is fixed, p(x=x*) = 1')
        # generate conditional mean
        else:
            if self.precompute:
                # find proper mu_1, mu_2, sgima_12_22_inv, and sigma_c
                index = self.mask_dict[str(mask)]  # find the right cache index
                fixed_indices = np.where(mask)
                variable_indices = np.where(mask == 0)
                a = x[fixed_indices]
                mu = self.mu_1[index] + np.matmul(
                    self.sigma_12_22_pinv[index], (a - self.mu_2[index])
                )
            else:
                fixed_indices = np.where(mask)
                variable_indices = np.where(mask == 0)
                a = x[fixed_indices]
                results = computemusigma(self.mu, self.sigma, mask)
                mu = results['mu_1'] + np.matmul(
                    results["sigma_12_22_pinv"], (a - results["mu_2"])
                )

            X = np.zeros_like(x)
            X[list(variable_indices[0])] = mu
            X[list(fixed_indices[0])] = x[fixed_indices]

        return X


class GaussianLinearRegression(CustomDataset):
    def __init__(
        self,
        mu,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigma=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples, num_classes=1)
        self.mu = mu
        self.dim = dim
        self.sigma = np.identity(self.dim) if sigma is None else sigma
        if rho:
            self.sigma += (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho
        if len(weight.shape) == 1:
            weight = np.expand_dims(weight, 1)
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MultivariateGaussian(
            mu=self.mu, sigma=self.sigma, dim=self.dim
        )

    def getdim(self):
        return self.dim

    def getweight(self):
        return self.weight

    def generatetarget(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        # print('generate y from x :')
        # print(X.shape)
        # print(len(self.weight.shape))
        Y = np.matmul(X, self.weight) + np.random.normal(
            scale=self.noise, size=(X.shape[0], 1)
        )
        Y -= np.mean(Y)
        Y /= np.std(Y)
        # print(Y.shape)
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


class GaussianPiecewiseConstantRegression(CustomDataset):
    def __init__(
        self,
        mu,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigma=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples, num_classes=1)
        self.mu = mu
        self.dim = dim
        self.sigma = sigma or np.identity(self.dim)
        if rho:
            self.sigma += (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MultivariateGaussian(
            mu=self.mu, sigma=self.sigma, dim=self.dim
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


class GaussianNonlinearAdditiveRegression(CustomDataset):
    """
    This class is a generalized version of the "Nonlinear Additive" dataset from the paper:
    "Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"
    The original dataset generates a 10 dimentional independent gaussian feature vector.
    Here we allow the features to be correlated.
    """

    def __init__(
        self,
        mu,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigma=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples, num_classes=1)
        self.mu = mu
        self.dim = dim
        self.sigma = sigma or np.identity(self.dim)
        if rho:
            self.sigma += (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MultivariateGaussian(
            mu=self.mu, sigma=self.sigma, dim=self.dim
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
            # print('nonlinear function: {}'.format(i))
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


class GaussianLinearBinary(CustomDataset):
    def __init__(
        self,
        mu,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigma=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples, num_classes=2)
        self.mu = mu
        self.dim = dim
        self.sigma = np.identity(self.dim) if sigma is None else sigma
        if rho:
            self.sigma += (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho
        if len(weight.shape) == 1:
            weight = np.expand_dims(weight, 1)
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MultivariateGaussian(
            mu=self.mu, sigma=self.sigma, dim=self.dim
        )

    def getdim(self):
        return self.dim

    def getweight(self):
        return self.weight

    def generatetarget(self, X):
        # print('generate y from x :')
        # print(X.shape)
        # print(len(self.weight.shape))
        if isinstance(X, pd.DataFrame):
            X = X.values
        Y = np.matmul(X, self.weight) + np.random.normal(
            scale=self.noise, size=(X.shape[0], 1)
        )
        Y[Y >= 0] = 1.0
        Y[Y < 0] = 0.0
        # print(Y[:10])
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


class GaussianPiecewiseConstantBinary(CustomDataset):
    def __init__(
        self,
        mu,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigma=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples, num_classes=2)
        self.mu = mu
        self.dim = dim
        self.sigma = sigma or np.identity(self.dim)
        if rho:
            self.sigma += (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MultivariateGaussian(
            mu=self.mu, sigma=self.sigma, dim=self.dim
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

        Y[Y >= 0] = 1.0
        Y[Y < 0] = 0.0
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


class GaussianNonlinearAdditiveBinary(CustomDataset):
    """
    This class is a generalized version of the "Nonlinear Additive" dataset from the paper:
    "Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"
    The original dataset generates a 10 dimentional independent gaussian feature vector.
    Here we allow the features to be correlated.
    """

    def __init__(
        self,
        mu,
        dim,
        weight,
        noise,
        num_train_samples=None,
        num_val_samples=None,
        sigma=None,
        rho=None,
    ):
        super().__init__(num_train_samples, num_val_samples, num_classes=2)
        self.mu = mu
        self.dim = dim
        self.sigma = sigma or np.identity(self.dim)
        if rho:
            self.sigma += (np.ones((self.dim, self.dim)) - np.identity(self.dim)) * rho
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(dim)
        self.generator = MultivariateGaussian(
            mu=self.mu, sigma=self.sigma, dim=self.dim
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
                p = -100 * np.sin(2 * x)
            elif i == 1:
                p = 2 * np.abs(x)
            elif i == 2:
                p = x
            elif i == 3:
                p = np.exp(-x)

            p = np.expand_dims(p, axis=1)
            # print('nonlinear function: {}'.format(i))
            # print(Y.shape)
            # print(p)
            Y = Y + p
        Y = Y - 2.4 + np.random.normal(scale=self.noise, size=(X.shape[0], 1))
        Y = np.exp(Y)
        Y = Y / (1 + Y)
        Y[Y >= 0.5] = 1.0
        Y[Y < 0.5] = 0.0
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


# # test
# N = 10
# D = 5
# rho = 0.0
# mean = np.zeros(D)
# weight = np.array([D-1-i for i in range(D)]).astype(float)
# cov = np.squeeze( (np.ones((D,D)) - np.identity(D)) * rho + np.identity(D) )
# means = [mean, mean]
# covs= [cov,cov]
# print(cov.shape)
# a = MultivariateGaussian(mean,cov,D)

# # a = MultivariateGaussian(mu=mean,sigma=cov,dim=5)
# x_s = np.array([0.1,0.2,0.3,100.0,-10.0])
# #x_s = np.array([0,0,0,0,0])
# x = a.generateconditional(mask=np.array([1,0,1,0,1])-0,
#                           x=x_s,
#                           n_sample=10)
# y = a.computeexpectation(mask=np.array([1,0,1,0,1])-0,
#                          x=np.array([0.1,0.2,0.3,100.0,-10.0]),)
# print(y)
# # m = np.mean(x,axis=0)
# # sd = np.std(x,axis=0)
# # print(x)
# # print(m)
# # print(sd)

# # b = GaussianLinearRegression(mu=mean,rho=0.2,dim=D,weight=weight,noise=0.01)

# # x = b.generator.computeexpectation(mask=np.array([1,0,1,0,1])-0,
# #                                    x=np.array([0.1,0.2,0.3,100.0,-10.0]),)
# print(x)

# #b = GaussianPiecewiseConstant(mu=mean,sigma=cov,dim=D,weight=weight,noise=0.01)
# #b = GaussianNonlinearAdditive(mu=mean,sigma=cov,dim=D,weight=weight,noise=0.01)
# #print(b.generate(n_sample=4))
