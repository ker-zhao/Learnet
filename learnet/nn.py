from learnet import type as ty
import numpy as np


def softmax(logits):
    exp_logits = ty.exp(logits)
    return ty.div(exp_logits, ty.broadcast(ty.reduce_sum(exp_logits, 1), exp_logits))


def cross_entropy(y_hat, y):
    x = ty.log(y_hat)
    x = ty.multiply(x, y)
    costs = ty.reduce_sum(x, axis=1)
    x = ty.sub(ty.broadcast(ty.constant(0), costs), costs)
    x = ty.mean(x)
    return x


def relu(x):
    return ty.relu(x)


def sigmoid(x):
    return ty.sigmoid(x)


def one_hot(y, n):
    y_one_hot = np.zeros((y.shape[0], n))
    y_one_hot[np.arange(y_one_hot.shape[0]), y] = 1
    return y_one_hot


def l2_regularizer(lambd=0.01):
    def inner(w, x):
        return ty.l2_regularizer(w, x, lambd)
    return inner
