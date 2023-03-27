from .ActivationFunction import ActivationFunction as AF
import numpy as np

class Sigmoid(AF):
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))