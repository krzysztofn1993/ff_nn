from .ActivationFunction import ActivationFunction as AF
import numpy as np

class ReLU(AF):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
