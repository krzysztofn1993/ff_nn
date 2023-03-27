from Layer import Layer
from ActivationFunctions import ReLU
from ActivationFunctions import Sigmoid
from ActivationFunctions import Softmax
from ActivationFunctions import ActivationFunction
from LossFunctions import CategoricalCrossEntropy as CCE
from LossFunctions import LossFunction
from typing import List

class NeuralNetwork:

    activation_function: ActivationFunction
    loss_function: LossFunction
    layers: List = []

    def __init__(self, configuration) -> None:
        # move to factory whole configuration?
        print(configuration)
        for layer_config in configuration['layer']:
            layer = Layer(layer_config[0], layer_config[1])
            self.layers.append(layer)

        self.activation_function = {
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'softmax': Softmax

        }[configuration['activation_function']]

        self.loss_function = {
            'CCE': CCE
        }[configuration['loss_function']]