import numpy as np
import random as r
from .Sigmoid import sigmoid, sigmoidDer
class Layer:
    def __init__(self, nbrNodes, prevNbrNodes,prevLayer = None, func = sigmoid, funcDer = sigmoidDer) -> None:
        self.nbrNodes = nbrNodes
        self.prevNbrNodes = prevNbrNodes
        self.nodes = np.zeros(nbrNodes)
        self.func = np.vectorize(func)
        self.funcDer = np.vectorize(funcDer)
        self.weights = np.array([[r.random()*2-1 for _ in range(self.prevNbrNodes)] for _ in range(self.nbrNodes)])
        self.bias = np.array([r.random()*2-1  for _ in range(self.nbrNodes)])
        self.prevLayer = prevLayer
        self.dWeights = np.zeros(self.weights.shape)
        self.dBias = np.zeros(self.bias.shape)
        self.layerNbr = -1
        #print(self.weights)

    def mul(self, vec):
        self.nodes = self.func(self.weights.dot(vec) + self.bias)
        #print(self.nodes)
        return self.nodes
    
    def backProp(self, dCdA):
        self.backPropWeights(dCdA)
        return self.backPropCalcdCdAj(dCdA)

    def backPropWeights(self, dCdA):
        dZ = np.array(list(map(self.funcDer, self.nodes))).reshape(self.nbrNodes,1)
        dW = (dZ * dCdA) * self.prevLayer.nodes
        #print("dZ: ", dZ)
        #print("dCdA: ", dCdA)
        #print("in: ", self.prevLayer.nodes)
        self.dWeights += dW


    def backPropCalcdCdAj(self, dCdA):
        dZ = np.array(list(map(self.funcDer, self.nodes))).reshape(self.nbrNodes,1)
        
        nextdCdA = (dZ * dCdA).reshape(1, dZ.size).dot(self.weights)
        return nextdCdA.reshape(nextdCdA.size,1)




        #colonVec = ((self.prevLayer.dCdAj)*(self.prevLayer.nodes.map(self.funcDer)))
        #colonVec = colonVec.reshape(colonVec.size,1)
        #colonVecdCdAj = np.dot(self.weights, colonVec)
        #self.dCdAj = colonVecdCdAj.reshape(1,colonVecdCdAj.size)

    def dCostLastLayer(self, correctArray):
        #print(self.nodes.size, correctArray.size)
        assert self.nodes.size == correctArray.size
        dCdAj = (self.nodes - correctArray) * -2
        return dCdAj.reshape(self.nbrNodes,1)
    
    def updateAfterBackProp(self):
        #print(self.dWeights)
        self.weights += self.dWeights/25
        #print("\n\nweights: ", self.weights)
        self.dWeights = np.zeros(self.weights.shape)
