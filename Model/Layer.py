import numpy as np
import random as r
from .Sigmoid import sigmoid, sigmoidDer
class Layer:
    def __init__(self, nbrNodes, prevNbrNodes,prevLayer = None, func = sigmoid, funcDer = sigmoidDer) -> None:
        self.nbrNodes = nbrNodes
        self.prevLayer = prevLayer
        self.nodes = np.zeros(nbrNodes)
        self.nodesNonSigmoid = np.zeros(nbrNodes)
        self.func = np.vectorize(func)
        self.funcDer = np.vectorize(funcDer)
        self.weights = np.array([[np.random.normal() for _ in range(prevNbrNodes)] for _ in range(self.nbrNodes)])
        self.bias = np.array([np.random.normal()  for _ in range(self.nbrNodes)])
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
        self.layerNbr = -1

    def mul(self, vec):
        self.nodesNonSigmoid = self.weights.dot(vec) + self.bias
        self.nodes = self.func(self.nodesNonSigmoid)
        return self.nodes
    
    def backProp(self, dCdA):
        dZ = np.array(list(map(self.funcDer, self.nodesNonSigmoid))).reshape(self.nbrNodes,1)
        dN = (dZ * dCdA)
        self.backPropWeights(dN)
        self.backPropBias(dN)
        return self.backPropCalcdCdAj(dN)

    def backPropWeights(self, dN):
        dW = dN.dot(np.array([self.prevLayer.nodes]))
        self.dWeights += dW

    def backPropBias(self, dN):
        self.dBias += dN.reshape(self.nbrNodes)

    def backPropCalcdCdAj(self, dN):
        nextdCdA = (dN.reshape(1, dN.size)).dot(self.weights)
        return nextdCdA.reshape(nextdCdA.size,1)

    def dCostLastLayer(self, correctArray):
        dCdAj = (self.nodes - correctArray) * 2
        return dCdAj.reshape(self.nbrNodes,1)
    
    def updateAfterBackProp(self, d):
        self.weights -= self.dWeights/d
        self.bias -= self.dBias/d
        self.resetDelta()

    def resetDelta(self):
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
