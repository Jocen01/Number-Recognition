from .Layer import Layer
from .StartLayer import StaryLayer
import numpy as np
class Ml:
    def __init__(self, layers) -> None:
        self.layers = [StaryLayer(layers[0])] + [Layer(i,layers[idx]) for idx ,i in enumerate(layers[1:])]
        for i in range(len(self.layers)-1):
            self.layers[i+1].prevLayer = self.layers[i]
        self.start = self.layers[0]
        self.result = None
        self.itr = 0

    def guess(self, arr):
        self.start.nodes = arr
        for l in self.layers:
            arr = l.mul(arr)
        self.result = arr
        return np.argmax(arr)
    
    def cost(self, correct):
        f = np.vectorize(lambda val: val**2)
        c = np.zeros(self.layers[-1].nbrNodes)
        c[correct] = 1
        return sum(f(self.result-c))
    
    def backProp(self, correct):
        self.itr += 1
        correctArr = np.zeros(self.layers[-1].nbrNodes)
        correctArr[correct] = 1
        dCdA = self.layers[-1].dCostLastLayer(correctArr)
        for lay in self.layers[::-1]:
            dCdA = lay.backProp(dCdA)
        if self.itr %100 == 0:
            for lay in self.layers:
                lay.updateAfterBackProp()
