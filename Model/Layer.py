import numpy as np
from .Sigmoid import sigmoid, sigmoidDer
class Layer:
    def __init__(self, nbrNodes, prevNbrNodes, prevLayer = None, func = sigmoid, funcDer = sigmoidDer) -> None:
        self.nbrNodes = nbrNodes
        self.prevLayer = prevLayer
        self.nodes = np.zeros((nbrNodes,1))
        self.func = np.vectorize(func)
        self.funcDer = np.vectorize(funcDer)
        self.weights = np.array([[np.random.normal() for _ in range(prevNbrNodes)] for _ in range(self.nbrNodes)])
        self.bias = np.array([[np.random.normal()]for _ in range(self.nbrNodes)])
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
        self.layerNbr = -1

    def mul(self, vec):
        self.nodes = self.weights @ vec.reshape(-1,1) + self.bias
        return self.func(self.nodes)
    
    def backProp(self, dCdA):
        dZ = self.funcDer(self.nodes)
        dN = (dZ * dCdA)
        self.backPropWeights(dN)
        self.backPropBias(dN)
        return self.backPropCalcdCdAj(dN)

    def backPropWeights(self, dN):
        dW = dN @ self.prevLayer.nodes.T
        self.dWeights += dW

    def backPropBias(self, dN):
        self.dBias += dN

    def backPropCalcdCdAj(self, dN):
        nextdCdA = self.weights.T @ dN
        return nextdCdA

    def dCostLastLayer(self, correctArray):
        assert self.nodes.shape == correctArray.shape
        dCdAj = (self.func(self.nodes) - correctArray) * 2
        return dCdAj
    
    def updateAfterBackProp(self, d):
        self.weights -= self.dWeights/d
        self.bias -= self.dBias/d
        self.resetDelta()

    def resetDelta(self):
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
