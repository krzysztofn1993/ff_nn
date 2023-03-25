import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer:
    def __init__(self,n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.10
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)
# print(X)
layer1 = Layer(2,3)
activation1 = ReLU()

layer2 = Layer(3,3)
activation2 = Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])
