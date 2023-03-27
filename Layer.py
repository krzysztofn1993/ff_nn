import numpy as np

class Layer:
    def __init__(self,n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.10
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases