from learnet import type as ty
import numpy as np


class Optimizer(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.cost = None

    def minimize(self, cost):
        self.cost = cost
        return ty.optimizer(self, self.cost)

    def gradient_check(self, feed_dict, epsilon=1e-7):
        nodes = self.cost.get_variable_nodes()
        grads = self.cost.gradients(nodes, feed_dict=feed_dict)
        grads_approx = []
        error_counter = 0
        for ni, node in enumerate(nodes):
            grads_approx.append(np.zeros_like(node.value))
            for i in range(node.value.shape[0]):
                for ii in range(node.value.shape[1]):
                    value_save = np.copy(node.value)
                    node.value[i, ii] = node.value[i, ii] + epsilon
                    cost_plus = self.cost.eval(feed_dict=feed_dict)
                    node.value[i, ii] = node.value[i, ii] - epsilon * 2
                    cost_minus = self.cost.eval(feed_dict=feed_dict)
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
        return self.cost.cache


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = []
        self.s = []
        self.v_correct = []
        self.s_correct = []
        self.t = 0

    def minimize(self, cost):
        self.cost = cost
        nodes = self.cost.get_variable_nodes()
        for node in nodes:
            self.v.append(np.zeros_like(node.value))
            self.s.append(np.zeros_like(node.value))
            self.v_correct.append(np.zeros_like(node.value))
            self.s_correct.append(np.zeros_like(node.value))
        return ty.optimizer(self, self.cost)

    def step(self):
        nodes = self.cost.get_variable_nodes()
        grads = self.cost.gradients(nodes)
        self.t += 1
        for i, node in enumerate(nodes):
            self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * grads[i]
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * (grads[i] ** 2)
            self.v_correct[i] = self.v[i] / (1 - (self.beta1 ** self.t))
            self.s_correct[i] = self.s[i] / (1 - (self.beta2 ** self.t))
            grad = self.v_correct[i] / (self.s_correct[i] ** 0.5 + self.epsilon)
            node.value -= grad * self.learning_rate
        return self.cost.cache
