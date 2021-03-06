import random
import time

from . import core
from . import optimizers
from . import nn
from .core import lib


opts = {
    "GD": optimizers.GradientDescent,
    "GradientDescent": optimizers.GradientDescent,
    "Adam": optimizers.Adam,
    "adam": optimizers.Adam,
}

losses = {
    "cross_entropy": nn.cross_entropy,
}


def accuracy(y_hat, y):
    y_hat = lib.np.argmax(y_hat, axis=1)
    y = lib.np.argmax(y, axis=1)
    correct = lib.np.sum(lib.np.equal(y_hat, y))
    return correct / y.shape[0]


class Model(object):
    def __init__(self):
        self.optimizer = None
        self.minimizer = None
        self.loss = None
        self.graph = None
        self.cost = None
        self.input = core.placeholder()
        self.y_placeholder = core.placeholder()

    def evaluate(self, x, y):
        print("Evaluate on {} samples".format(x.shape[0]))
        loss = self.cost.eval(feed_dict={self.input: x, self.y_placeholder: y})[0, 0]
        y_hat = self.graph.cache
        acc = accuracy(y_hat, y)
        print("Loss: {}, accuracy: {}.".format(loss, acc))
        return [loss, acc]

    def predict(self, x):
        y_hat = self.graph.eval(feed_dict={self.input: x})
        return y_hat

    def fit(self, x, y, batch_size=32, epochs=1, shuffle=True, verbose=2):
        print("Train on {} samples".format(x.shape[0]))
        if shuffle:
            indexes = [i for i in range(x.shape[0])]
            random.shuffle(indexes)
            x, y = x[indexes, ...], y[indexes, ...]
        n_batches = int(x.shape[0] / batch_size)
        for epoch in range(epochs):
            time_start = time.time()
            print("Epoch {}/{}".format(epoch + 1, epochs))
            epoch_loss = 0
            epoch_accuracy = 0
            for nb in range(n_batches):
                mini_x = x[batch_size * nb:batch_size * (nb + 1), :]
                mini_y = y[batch_size * nb:batch_size * (nb + 1), :]
                loss = self.minimizer.eval(feed_dict={self.input: mini_x, self.y_placeholder: mini_y})[0, 0]
                acc = accuracy(self.graph.cache, mini_y)
                epoch_loss = epoch_loss + loss
                epoch_accuracy = epoch_accuracy + acc
                if verbose >= 2:
                    print("Epoch: {}, loss: {}, accuracy: {}.".format(epoch, loss, acc))
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            time_used = time.time() - time_start
            time_str = "{}m {:.2f}s".format(int(time_used / 60), time_used % 60)
            if verbose >= 1:
                # print("=" * 100)
                # print("==== Epoch {} finished, loss: {}, accuracy: {}.".format(epoch, epoch_loss, epoch_accuracy))
                print("Time used: {},  loss: {:.6f}, accuracy: {:.4f}.".format(
                    time_str, float(epoch_loss), float(epoch_accuracy)))

    def grad_check(self, x, y):
        self.optimizer.gradient_check(feed_dict={self.input: x, self.y_placeholder: y})


class Sequential(Model):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers if layers else []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = opts[optimizer]() if isinstance(optimizer, str) else optimizer
        self.loss = losses[loss] if isinstance(loss, str) else loss
        reg_cost = None

        self.graph = self.input
        prev_output_dims = self.layers[0].input_dims
        for layer in self.layers:
            self.graph = layer.get_graph(self.graph, prev_output_dims)
            prev_output_dims = layer.output_units
            if layer.kernel_regularizer:
                if reg_cost:
                    reg_cost = core.add(reg_cost, layer.kernel_regularizer(layer.w, self.graph))
                else:
                    reg_cost = layer.kernel_regularizer(layer.w, self.graph)
        self.cost = self.loss(self.graph, self.y_placeholder)
        if reg_cost:
            self.cost = core.add(self.cost, reg_cost)
        self.minimizer = self.optimizer.minimize(self.cost)
