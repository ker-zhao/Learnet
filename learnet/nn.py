import learnet as ln
import numpy as np


def softmax(logits):
    exp_logits = ln.exp(logits)
    return ln.div(exp_logits, ln.broadcast(ln.reduce_sum(ln.exp(logits), 0), exp_logits))


def cross_entropy(y_hat, y):
    x = ln.log(y_hat)
    x = ln.multiply(x, y)
    costs = ln.reduce_sum(x, axis=0)
    x = ln.sub(ln.broadcast(ln.constant(0), costs), costs)
    x = ln.mean(x)
    return x


def one_hot(y, n):
    y_one_hot = np.zeros((y.shape[0], n))
    y_one_hot[np.arange(y_one_hot.shape[0]), y] = 1
    return y_one_hot
