from . import core
from .core import lib


def softmax(logits):
    exp_logits = core.exp(logits)
    return core.div(exp_logits, core.broadcast(core.reduce_sum(exp_logits, 1), exp_logits))


def cross_entropy(y_hat, y):
    x = core.log(y_hat)
    x = core.multiply(x, y)
    costs = core.reduce_sum(x, axis=1)
    x = core.mean(core.negative(costs))
    return x


def relu(x):
    return core.relu(x)


def sigmoid(x):
    return core.sigmoid(x)


def one_hot(y, n):
    y_one_hot = lib.np.zeros((y.shape[0], n))
    y_one_hot[lib.np.arange(y_one_hot.shape[0]), y] = 1
    return y_one_hot


def l2_regularizer(lambd=0.01):
    def inner(w, x):
        return core.l2_regularizer(w, x, lambd)
    return inner
