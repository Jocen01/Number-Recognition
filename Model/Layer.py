import numpy as np
import random as r
from .Sigmoid import sigmoid, sigmoidDer
class Layer:
    def __init__(self, nbrNodes, prevNbrNodes,prevLayer = None, func = sigmoid, funcDer = sigmoidDer) -> None:
        self.nbrNodes = nbrNodes
        self.prevLayer = prevLayer
        self.nodes = np.zeros(nbrNodes)
        self.func = np.vectorize(func)
        self.funcDer = np.vectorize(funcDer)
        self.weights = np.array([[r.random()*2-1 for _ in range(prevNbrNodes)] for _ in range(self.nbrNodes)])
        self.bias = np.array([r.random()*2-1  for _ in range(self.nbrNodes)])
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
        self.layerNbr = -1

    def mul(self, vec):
        self.nodes = self.func(self.weights.dot(vec) + self.bias)
        return self.nodes
    
    def backProp(self, dCdA):
        dZ = np.array(list(map(self.funcDer, self.nodes))).reshape(self.nbrNodes,1)
        dN = (dZ * dCdA)
        self.backPropWeights(dN)
        self.backPropBias(dCdA)
        return self.backPropCalcdCdAj(dCdA)

    def backPropWeights(self, dN):
        dW = dN * self.prevLayer.nodes
        self.dWeights += dW

    def backPropBias(self, dN):
        self.dBias += dN.reshape(self.nbrNodes)

    def backPropCalcdCdAj(self, dN):
        nextdCdA = dN.reshape(1, dN.size).dot(self.weights)
        return nextdCdA.reshape(nextdCdA.size,1)

    def dCostLastLayer(self, correctArray):
        assert self.nodes.size == correctArray.size
        dCdAj = (self.nodes - correctArray) * -2
        return dCdAj.reshape(self.nbrNodes,1)
    
    def updateAfterBackProp(self):
        self.weights += self.dWeights/25
        self.bias += self.dBias/25
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
