from .Layer import Layer
from .StartLayer import StaryLayer
from .Utilities import *
import numpy as np
import random
import time
class Ml:
    def __init__(self, layers) -> None:
        self.layers = [StaryLayer(layers[0])] + [Layer(i,layers[idx]) for idx ,i in enumerate(layers[1:])]
        for i in range(len(self.layers)-1):
            self.layers[i+1].prevLayer = self.layers[i]
        self.start = self.layers[0]
        self.result = None

    def guess(self, arr):
        self.start.nodes = arr
        for l in self.layers[1:]:
            arr = l.mul(arr)
        self.result = arr
        return np.argmax(arr)
    
    def cost(self, correct):
        f = np.vectorize(lambda val: val**2)
        c = np.zeros(self.layers[-1].nbrNodes)
        c[correct] = 1
        return sum(f(self.result-c))
    
    def backProp(self, correct):
        correctArr = np.zeros(self.layers[-1].nbrNodes)
        correctArr[correct] = 1
        dCdA = self.layers[-1].dCostLastLayer(correctArr)
        for lay in self.layers[:0:-1]:
            dCdA = lay.backProp(dCdA)

    def resetDelta(self):
        for lay in self.layers[1:]:
            lay.resetDelta()
    
    def update(self, d):
        for lay in self.layers[1:]:
            lay.updateAfterBackProp(d)

    def runEpoch(self, trainIn, trainAns, sizeBatch, gradientFactor):
        normalize = max(np.ndarray.flatten(trainIn[0]))
        print(sizeBatch/gradientFactor, sizeBatch, gradientFactor)
        l = [i for i in range(len(trainIn))]
        random.shuffle(l)
        for idx,i in enumerate(l):
            data, correct = trainIn[i], trainAns[i]
            self.guess(np.ndarray.flatten(data)/normalize)
            self.backProp(correct)
            if (idx+1)%sizeBatch == 0:
                self.update(sizeBatch/gradientFactor)

    def test(self, testIn, testAns):
        correct = 0
        cost = 0
        for i in range(len(testIn)):
            if self.guess(np.ndarray.flatten(testIn[i])) == testAns[i]:
                correct += 1
            cost +=self.cost(testAns[i])
        return correct/len(testIn), cost/len(testIn)
    
    def testShow(self, testIn, testAns):
        correct = 0
        for i in range(len(testIn)):
            guess = self.guess(np.ndarray.flatten(testIn[i]))
            if guess == testAns[i]:
                correct += 1
            
            printImage(testIn[i])
            print("AI guessed", guess, "correct answer was", testAns[i])
            input()
        return correct/len(testIn)
    
    def train(self, trainIn, trainAns, epochs):
        for i in range(epochs):
            self.resetDelta()
            t1 = time.time()
            self.runEpoch(trainIn, trainAns,100, max(1, 4/(i+1)))
            print("Epoch ran in: ", round(time.time()-t1,4), " seconds")
            t2 = time.time()
            score = self.test(trainIn[:10000],trainAns[:10000])
            print("Score for the network after ", i+1, " ephochs is ", score)
            print("Score calculated on 10000 examples from the training data in ", round(time.time()-t2,4), " seconds")
