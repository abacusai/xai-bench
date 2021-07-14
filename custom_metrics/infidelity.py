"""
Implementation of https://arxiv.org/pdf/1901.09392.pdf, based on https://github.com/chihkuanyeh/saliency_evaluation/blob/master/infid_sen_utils.py
"""
import numpy as np

def get_exp(ind, exp):
    return (exp[ind.astype(int)])

class Infidelity:
    def __init__(self, model, trained_model, dataset=None, **kwargs):
        self.model = model
        self.trained_model = trained_model
        self.dataset = dataset
    
    def set_zero_infid(self, array, size, point, pert):
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])
    
    def evaluate(self, X, y, feature_weights, ground_truth_weights, avg=True, X_train=None, y_train=None, n_sample=100, X_train_feature_weights=None):
        X = X.values
        num_datapoints, num_features = X.shape
        absolute_weights = abs(feature_weights)
        infids = []

        for i in range(num_datapoints):
            num_reps = 1000
            x_orig = np.tile(X[i], [num_reps, 1])
            x = X[i]
            expl = feature_weights[i]
            expl_copy = np.copy(expl)
            val = np.apply_along_axis(self.set_zero_infid, 1, x_orig, num_features, num_features, pert="Gaussian")
            x_ptb, ind, rand = val[:, :num_features], val[:, num_features: 2*num_features], val[:, 2*num_features: 3*num_features]
            exp_sum = np.sum(rand*np.apply_along_axis(get_exp, 1, ind, expl_copy), axis=1)
            ks = np.ones(num_reps)
            pdt = self.trained_model.predict([x])
            pdt_ptb = self.trained_model.predict(x_ptb)
            pdt_diff = pdt - pdt_ptb

            beta = np.mean(ks*pdt_diff*exp_sum) / np.mean(ks*exp_sum*exp_sum)
            exp_sum *= beta
            infid = np.mean(ks*np.square(pdt_diff-exp_sum)) / np.mean(ks)
            infids.append(infid)
        
        return np.mean(infids)