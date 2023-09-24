import numpy as np
import random as r
from .Sigmoid import sigmoid
class Layer:
    def __init__(self, nbrNodes, prevNbrNodes, func = sigmoid) -> None:
        self.nbrNodes = nbrNodes
        self.prevNbrNodes = prevNbrNodes
        self.func = np.vectorize(func)
        self.weights = np.array([[r.random()*2-1 for _ in range(self.prevNbrNodes)] for _ in range(self.nbrNodes)])
        self.bias = np.array([r.random()*2-1  for _ in range(self.nbrNodes)])

    def mul(self, vec):
        
        res = self.func(self.weights.dot(vec) + self.bias)
        #print(res)
        return res