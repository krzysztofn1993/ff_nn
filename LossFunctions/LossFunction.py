import numpy as np

class LossFunction:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)

        return data_loss