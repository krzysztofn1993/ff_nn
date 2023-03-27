from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @property
    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass