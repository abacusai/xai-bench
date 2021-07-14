from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd

# import cPickle as pkl
from collections import defaultdict
import re

# from bs4 import BeautifulSoup
import sys
import os
import glob
import time
import copy
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer

# from make_data import generate_data
import json
import random
from keras import optimizers

BATCH_SIZE = 1000
# np.random.seed(0)
#tf.random.set_seed(0)
#tf.random.set_random_seed(0)
# random.seed(0)


def create_data(datatype, n=1000):
    """
    Create train and validation datasets.

    """
    x_train, y_train, _ = generate_data(n=n, datatype=datatype, seed=0)
    x_val, y_val, datatypes_val = generate_data(n=10 ** 5, datatype=datatype, seed=1)

    input_shape = x_train.shape[1]

    return x_train, y_train, x_val, y_val, datatypes_val, input_shape


def create_rank(scores, k):
    """
    Compute rank of each feature based on weight.

    """
    scores = abs(scores)
    n, d = scores.shape
    ranks = []
    for i, score in enumerate(scores):
        # Random permutation to avoid bias due to equal weights.
        idx = np.random.permutation(d)
        permutated_weights = score[idx]
        permutated_rank = (-permutated_weights).argsort().argsort() + 1
        rank = permutated_rank[np.argsort(idx)]

        ranks.append(rank)
    print("n: {}, d: {}, scores: {}, ranks: {}".format(n, d, scores[0:4], ranks[0:4]))
    return np.array(ranks)


def compute_median_rank(scores, k, datatype_val=None):
    ranks = create_rank(scores, k)
    if datatype_val is None:
        median_ranks = np.median(ranks[:, :k], axis=1)
        print(median_ranks)
    else:
        datatype_val = datatype_val[: len(scores)]
        median_ranks1 = np.median(
            ranks[datatype_val == "orange_skin", :][:, np.array([0, 1, 2, 3, 9])],
            axis=1,
        )
        median_ranks2 = np.median(
            ranks[datatype_val == "nonlinear_additive", :][
                :, np.array([4, 5, 6, 7, 9])
            ],
            axis=1,
        )
        median_ranks = np.concatenate((median_ranks1, median_ranks2), 0)
    return median_ranks


class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables.

    """

    def __init__(self, tau0, k, **kwargs):
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d]
        logits_ = K.expand_dims(logits, -2)  # [BATCH_SIZE, 1, d]

        batch_size = tf.shape(logits_)[0]
        d = tf.shape(logits_)[2]
        uniform = tf.random.uniform(
            shape=(batch_size, self.k, d),
            minval=np.finfo(tf.float32.as_numpy_dtype).tiny,
            maxval=1.0,
        )

        gumbel = -K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis=1)

        # Explanation Stage output.
        threshold = tf.expand_dims(
            tf.nn.top_k(logits, self.k, sorted=True)[0][:, -1], -1
        )
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(samples, discrete_logits)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'tau0': self.tau0,
            'k': self.k
        })
        return config

def get_filepath():
    currentLogs = glob.glob(f"results/saved_models/*-L2X.hdf5")
    numList = [0]
    for i in currentLogs:
        i = os.path.splitext(i)[0]
        try:
            num = re.findall("[0-9]+$", i)[0]
            numList.append(int(num))
        except IndexError:
            pass
    numList = sorted(numList)
    newNum = numList[-1] + 1
    return f"results/saved_models/{newNum}-L2X.hdf5"

def buildmodel(x, y, k, input_shape, n_class=2):
    # n_class = 1: regression task
    # n_class > 1: classfication task (not implemented yet)
    if n_class == 1:
        loss = "mse"
        monitor = "val_loss"
        final_activation = "linear"
        save_mode = 'min'
    elif n_class > 1:
        print('shape')
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1) 
        print(y.shape)
        if y.shape[1] == 1 and n_class == 2:
            y_new = np.zeros((y.shape[0],n_class))
            y_new[:,0] = copy.deepcopy(np.squeeze(y))
            y_new[:,1] = 1 - np.squeeze(y)
            y = copy.deepcopy(y_new)
            # for i in range(y_new.shape[0]):
            #     print(y_new[i,:])
        loss = "categorical_crossentropy"
        monitor = "val_acc"
        final_activation = "softmax"
        save_mode = 'max'
    st1 = time.time()
    st2 = st1
    activation = "selu"
    l2 = 1e-3  # default 1e-3
    # P(S|X)
    model_input = Input(shape=(input_shape,), dtype="float32")

    net = Dense(
        100,
        activation=activation,
        name="s/dense1",
        kernel_regularizer=regularizers.l2(l2),
    )(model_input)
    net = Dense(
        100,
        activation=activation,
        name="s/dense2",
        kernel_regularizer=regularizers.l2(l2),
    )(net)

    # A tensor of shape, [batch_size, max_sents, 100]
    logits = Dense(input_shape)(net)
    # [BATCH_SIZE, max_sents, 1]
    tau = 0.1
    samples = Sample_Concrete(tau, k, name="sample")(logits)

    # q(X_S)
    new_model_input = Multiply()([model_input, samples])
    net = Dense(
        200,
        activation=activation,
        name="dense1",
        kernel_regularizer=regularizers.l2(l2),
    )(new_model_input)
    net = BatchNormalization()(net)  # Add batchnorm for stability.
    net = Dense(
        200,
        activation=activation,
        name="dense2",
        kernel_regularizer=regularizers.l2(l2),
    )(net)
    net = BatchNormalization()(net)

    preds = Dense(
        n_class,
        activation=final_activation,
        name="dense4",
        kernel_regularizer=regularizers.l2(l2),
    )(net)
    model = Model(model_input, preds)
    pred_model = Model(model_input, samples)

    adam = optimizers.Adam(lr=1e-3)
    model.compile(loss=loss, optimizer=adam, metrics=["acc", "mean_squared_error"])

    # filepath = get_filepath()
    # filepath = "{}-L2X.hdf5".format(
    #     k
    # )  # Yang: hacky way to get the model to store with some name
    # checkpoint = ModelCheckpoint(
    #     filepath, monitor=monitor, verbose=1, save_best_only=True, mode=save_mode
    # )
    callbacks_list = []
    # print("start training, k: {}, final nonlinearity: {}".format(k, final_activation))
    model.fit(
        x,
        y,
        validation_data=(x, y),
        callbacks=callbacks_list,
        epochs=1,
        batch_size=BATCH_SIZE,
    )

    pred_model.compile(
        loss=None, optimizer="rmsprop", metrics=["acc", "mean_squared_error"]
    )

    return model, pred_model


class L2X:
    """
    Rhe original l2x out put a vector of binary feature importance values for a
    chosen k, with k being the number of important features the user thinks is in
    the dataset. First of all user has to choose it, it's unclear how you would choose
    the best k. Second, since all important values have weights of 1, it is
    impossible to rank them.

    To overcome this, Yang proposes to run l2x for k = 1,2,...,M and add feature
    importance values. This way the most important feature will have a final importance
    value of M, because it will be 1 in each run.

    """

    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X.values
        self.M = X.shape[1]
        if 'batch_size' in kwargs:
            BATCH_SIZE = kwargs['batch_size']
        # set up models with k = 1,2,3,..., M
        self.models = []
        self.pred_models = []
        self.Y = self.f(X)
        # print("X shape", X.shape)
        for k in range(1, self.M + 1):
            model, pred_model = buildmodel(self.X, self.Y, k, self.M)
            self.models.append(model)
            self.pred_models.append(pred_model)

    def explain(self, x):
        weights = np.zeros_like(x)
        #x = np.ones_like(x)
        self.expected_values = np.ones((x.shape[0], 1)) * np.mean(self.Y)
        for i in range(len(self.models)):
            # if i == 3:
                weights = weights + self.pred_models[i].predict(
                    x, verbose=1, batch_size=BATCH_SIZE
                )
        # normalize
        weights = weights / np.expand_dims(np.sum(weights, axis=1), 1)
                # print('k:', i+1)
        #print(weights[:10])
                # print(np.sum(weights,axis=0))
                # median_ranks = compute_median_rank(weights,4)
                # print(np.mean(median_ranks))
        return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datatype",
        type=str,
        choices=["orange_skin", "XOR", "nonlinear_additive", "switch"],
        default="orange_skin",
    )
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    median_ranks, exp_time, train_time = L2X(datatype=args.datatype, train=args.train)
    output = "datatype:{}, mean:{}, sd:{}, train time:{}s, explain time:{}s \n".format(
        args.datatype, np.mean(median_ranks), np.std(median_ranks), train_time, exp_time
    )

    print(output)
