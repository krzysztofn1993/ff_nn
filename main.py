import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

inputs = [ 0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []


class Layer:
    def __init__(self,n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.10
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer(2, 5)
activation1 = ReLU()

layer1.forward(X)


activation1.forward(layer1.output)
print(activation1.output)
