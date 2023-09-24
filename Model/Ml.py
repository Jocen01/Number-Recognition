from .Layer import Layer
import numpy as np
class Ml:
    def __init__(self, layers) -> None:
        self.layers = [Layer(i,layers[idx]) for idx ,i in enumerate(layers[1:])]

    def guess(self, arr):
        for l in self.layers:
            arr = l.mul(arr)
        return np.argmax(arr)