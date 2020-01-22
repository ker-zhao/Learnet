from learnet import type as ty
import numpy as np


class Optimizer(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.cost = None

    def minimize(self, cost):
        self.cost = cost
        return ty.optimizer(self)

    def gradient_check(self, epsilon=1e-7):
        nodes = self.cost.get_variable_nodes()
        grads = self.cost.gradients(nodes)
        grads_approx = []
        error_counter = 0
        for ni, node in enumerate(nodes):
            grads_approx.append(np.zeros_like(node.value))
            for i in range(node.value.shape[0]):
                for ii in range(node.value.shape[1]):
                    value_save = np.copy(node.value)
                    node.value[i, ii] = node.value[i, ii] + epsilon
                    cost_plus = self.cost.eval()
                    node.value[i, ii] = node.value[i, ii] - epsilon * 2
                    cost_minus = self.cost.eval()
                    grad_approx = (cost_plus - cost_minus) / (epsilon * 2)
                    grads_approx[ni][i, ii] = grad_approx
                    grad = grads[ni][i, ii]
                    diff = abs(grad_approx - grad)
                    if diff > epsilon:
                        print("Error: gradient_check: grads: {}, grad_approx: {}, diff: {}".format(
                            grads[ni][i, ii], grad_approx, abs(grads_approx[ni][i, ii] - grads[ni][i, ii])))
                        error_counter += 1
                    node.value = np.copy(value_save)
        print("Gradient checking pass. Errors: {}".format(error_counter))


class GradientDescent(Optimizer):
    def step(self):
        nodes = self.cost.get_variable_nodes()
        grads = self.cost.gradients(nodes)
        for i, node in enumerate(nodes):
            node.value -= grads[i] * self.learning_rate
