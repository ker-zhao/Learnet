import numpy as np
from learnet import type as ty


def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    y = np.argmax(y, axis=1)
    correct = np.sum(np.equal(y_hat, y))
    return correct / y.shape[0] * 100.00


class Model(object):
    def __init__(self):
        self.optimizer = None
        self.minimizer = None
        self.loss = None
        self.graph = None
        self.cost = None
        self.input = ty.placeholder()
        self.y_placeholder = ty.placeholder()

    def fit(self, x, y, batch_size=32, epochs=1, shuffle=True):
        n_batches = int(x.shape[0] / batch_size)
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            for nb in range(n_batches):
                mini_x = x[batch_size * nb:batch_size * (nb + 1), :]
                mini_y = y[batch_size * nb:batch_size * (nb + 1), :]
                self.minimizer.eval(feed_dict={self.input: mini_x, self.y_placeholder: mini_y})

                loss = self.cost.eval()[0, 0]
                acc = accuracy(self.graph.eval(), mini_y)
                epoch_loss = epoch_loss + loss
                epoch_accuracy = epoch_accuracy + acc
                print("Epoch: {}, loss: {}, accuracy: {}.".format(epoch, loss, acc))

            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            print("-" * 100)
            print("Epoch {} finished, loss: {}, accuracy: {}.".format(epoch, epoch_loss, epoch_accuracy))

    def grad_check(self, x, y):
        self.optimizer.gradient_check(feed_dict={self.input: x, self.y_placeholder: y})


class Sequential(Model):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

        self.graph = self.input
        prev_output_dims = self.layers[0].input_dims
        for layer in self.layers:
            self.graph = layer.get_graph(self.graph, prev_output_dims)
            prev_output_dims = layer.output_units

        self.cost = self.loss(self.graph, self.y_placeholder)
        self.minimizer = self.optimizer.minimize(self.cost)
