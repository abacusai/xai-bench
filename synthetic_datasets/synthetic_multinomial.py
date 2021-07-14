import copy
from synthetic_datasets.custom_dataset import CustomDataset
import numpy as np
import pandas as pd


class MultivariateMultinomial:
    def __init__(self, dim, n_param, p_param):
        self.dim = dim
        self.n_param = n_param
        self.p_param = p_param
    
    def generateconditional(self, mask, x, n_sample):
        
        # assert(type(x) == np.ndarray) 
        # assert(len(x.shape) == 1)
        # assert(type(mask[0]) == bool)
        # assert(type(n_sample) == int) 
        # assert(type(self.n_param) == int) 
        # assert(type(self.p_param) == np.ndarray) 
        # assert(len(self.p_param.shape) == 1)
        
        # Access conditional values
        if not mask.any():
            return np.random.multinomial(self.n_param, self.p_param, n_sample)

        np_mask = np.array(mask)
        cond_x = x[np_mask]
        cond_p_param = self.p_param[np_mask]

        # Sample from conditional multinomial
        new_n_param = self.n_param - np.sum(cond_x)
        new_p_param = self.p_param[~np_mask] / (1 - np.sum(cond_p_param))
        samp_arr = np.random.multinomial(new_n_param, new_p_param, n_sample)

        # format output samples into list
        x_new_list = [copy.deepcopy(x) for _ in samp_arr]
        for x, samp in zip(x_new_list, samp_arr):
            x[~np_mask] = samp

        return np.array(x_new_list)


class MultinomialLinearRegression(CustomDataset):
    def __init__(self, dim, n_param, p_param, weight, noise, num_train_samples=None, num_val_samples=None, num_classes=None):
        super().__init__(num_train_samples, num_val_samples, num_classes=num_classes)
        self.dim = dim
        self.n_param = n_param
        self.p_param = p_param
        self.weight = weight
        self.noise = noise
        self.default_mask = np.zeros(self.dim).astype(bool)
        self.generator = MultivariateMultinomial(self.dim, self.n_param, self.p_param)
    
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
            mask = mask.astype(np.bool)

        if x is None:
            x = self.default_mask

        X = self.generator.generateconditional(mask=mask, x=x, n_sample=n_sample)
        Y = self.generatetarget(X)

        return X, Y


# a = MultinomialLinearRegression(5, 20, np.array([0.2, 0.3, 0.1, 0.1, 0.3]), np.array([4, 3, 2, 1, 0]), 0.01)

