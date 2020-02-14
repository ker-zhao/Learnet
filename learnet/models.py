
from learnet import type as ty


class Model(object):
    def __init__(self):
        self.optimizer = None
        self.minimizer = None
        self.loss = None
        self.graph = None
        self.cost = None
        self.input = ty.placeholder()
        self.y_placeholder = ty.placeholder()

    def fit(self, x, y, batch_size=32, epochs=1):
        for i in range(epochs):
            self.minimizer.eval(feed_dict={self.input: x, self.y_placeholder: y})


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
