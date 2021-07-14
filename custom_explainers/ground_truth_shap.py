import scipy.special
import numpy as np
import itertools
import copy
from tqdm import tqdm


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 1000000  # approximation of inf with some large weight
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


class BruteForceKernelShap:
    def __init__(self, f, X, n=1000, **kwargs):
        self.X = X
        self.f = f
        self.n = n
        self.dim = X.shape[1]
        self.reference = np.mean(X, axis=0)

    def explain_x(self, x):

        X = np.zeros((2 ** self.dim, self.dim))
        # X[:,-1] = 1
        weights = np.zeros(2 ** self.dim)
        V = np.zeros((2 ** self.dim, self.dim))
        for i in range(2 ** self.dim):
            V[i, :] = self.reference  # this works only with independence assumption
        
        y = np.zeros(2 ** self.dim)
        for i, s in enumerate(powerset(range(self.dim))):
            s = list(s)
            V[i, s] = x[s]
            X[i, s] = 1
            weights[i] = shapley_kernel(self.dim, len(s))
            x_s = np.copy(self.X[:self.n])
            x_s[:, s] = x[s]
            y_temp = self.f(x_s)
            y[i] = np.mean(y_temp)

        # y = self.f(V)
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        coefs = np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))
        expectation = y[0]
        return expectation, coefs

    def explain(self, X):
        self.expected_values = np.zeros((X.shape[0], 1))
        shap_values = np.zeros((X.shape[0], self.dim))
        for idx, x in tqdm(enumerate(X.values)):
            self.expected_values[idx], shap_values[idx] = self.explain_x(x)
        return shap_values


class GroundTruthShap:
    def __init__(
        self,
        f=None,  # model to explain, if None then explain dataset
        dataset=None,  # dataset to explain
        n=20000,  # number of samples to estimate E(f(x_1|x_2 = x* ))
    ):

        self.dataset = dataset
        self.f = f
        self.n = n
        self.dim = self.dataset.getdim()
        assert dataset is not None
        if f is None:
            print("No model passed, explaining dataset!")

    def explain(self, x):

        X = np.zeros((2 ** self.dim, self.dim))
        # X[:,-1] = 1
        weights = np.zeros(2 ** self.dim)
        V = np.zeros((2 ** self.dim, self.dim))
        # for i in range(2**self.dim):
        #    V[i,:] = reference #this works only with independence assumption

        ws = {}
        y = np.zeros((2 ** self.dim, 1))
        for i, s in enumerate(powerset(range(self.dim))):
            s = list(s)
            V[i, s] = x[s]
            X[i, s] = 1
            x_s, y_s = self.dataset.generate(mask=X[i, :], x=V[i, :], n_sample=self.n)
            if self.f is None:
                y[i] = float(np.mean(y_s))
            else:  # this might need proper formating
                y_temp = self.f(
                    x_s
                )  # pass conditional x to model to make batch predictions
                y[i] = float(np.mean(y_temp))  # compute the expectation
            ws[len(s)] = ws.get(len(s), 0) + shapley_kernel(self.dim, len(s))
            weights[i] = shapley_kernel(self.dim, len(s))

        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        coefs = np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))
        expectation = y[0]
        return expectation, coefs


# # test
# N = 10
# D = 6
# rho = 0.5
# mean = np.zeros(D)
# weight = np.array([D-1-i for i in range(D)]).astype(float)
# weight = np.array([3,2,1,0,0,0])
# #weight = weight / np.sum(weight)
# #weight = np.array([1 for i in range(D)]).astype(float)
# #print(weight)
# cov = (np.ones((D,D)) - np.identity(D)) * rho + np.identity(D)
# a = MultivariateGaussian(mu=mean,sigma=cov,dim=5)

# x = a.generateconditional(mask=np.array([1,1,1,1,1])-0,
#                           x=np.array([0.1,0.2,0.3,100.0,-10.0]),
#                           n_sample=1000)

# m = np.mean(x,axis=0)
# sd = np.std(x,axis=0)
# #print(x)
# #print(m)
# #print(sd)
# #print(mean)
# #print(cov)
# b = GaussianLinearDataset(mu=mean,sigma=cov,dim=D,weight=weight,noise=0.00)
# #b = GaussianPiecewiseConstant(mu=mean,sigma=cov,dim=D,weight=weight,noise=0.01)
# x,y = b.generate(n_sample=5)
# #print(x,y)
# c = GroundTruthShap(dataset=b,n=2000)
# d = KernelShap(reference=np.ones_like(x[0])*0,dataset=b,f=None)
# exp = c.explain(x[0])
# exp2 = c.explain(np.ones_like(x[0])*1)
# #exp3 = c.explain(np.ones_like(x[0])*-1)
# exp4 = d.explain(np.ones_like(x[0])*1)
# #print(exp)
# print('linear weights:')
# print(weight)
# print('ground truth explainer coefficients:')
# print(exp2)
# print('bruteforce explainer coefficients:')
# print(exp4)
# #b = GaussianPiecewiseConstant(mu=mean,sigma=cov,dim=D,weight=weight,noise=0.01)
# #b = GaussianNonlinearAdditive(mu=mean,sigma=cov,dim=D,weight=weight,noise=0.01)
# #print(b.generate(n_sample=4))
