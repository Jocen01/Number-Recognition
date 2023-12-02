from .Layer import Layer
from .StartLayer import StaryLayer
from .Utilities import *
import numpy as np
import random
import time
class Ml:
    def __init__(self, layers, batchSize=100, learnRate=1, costFunc=None) -> None:
        self.layers = [StaryLayer(layers[0])] + [Layer(i,layers[idx]) for idx ,i in enumerate(layers[1:])]
        for i in range(len(self.layers)-1):
            self.layers[i+1].prevLayer = self.layers[i]
        self.start = self.layers[0]
        self.batchSize = batchSize
        self.lr = learnRate
        self.constFunc = np.vectorize(lambda val: val**2) if not costFunc else costFunc
        self.result = None

    def guess(self, arr):
        self.start.nodes = arr.reshape(-1,1)
        for l in self.layers[1:]:
            arr = l.mul(arr)
        self.result = arr
        return np.argmax(arr)
    
    def cost(self, correct):
        c = np.zeros((self.layers[-1].nbrNodes,1))
        c[correct,0] = 1
        return sum(self.constFunc(self.result-c))[0]
    
    def backProp(self, correct):
        correctArr = np.zeros((self.layers[-1].nbrNodes,1))
        correctArr[correct,0] = 1
        dCdA = self.layers[-1].dCostLastLayer(correctArr)
        for lay in self.layers[:0:-1]:
            dCdA = lay.backProp(dCdA)

    def resetDelta(self):
        for lay in self.layers[1:]:
            lay.resetDelta()
    
    def update(self, d):
        for lay in self.layers[1:]:
            lay.updateAfterBackProp(d)

    def runEpoch(self, trainIn, trainAns):
        n = max(np.ndarray.flatten(trainIn[0]))
        data = list(zip(trainIn, trainAns))
        random.shuffle(data)
        for idx,(img,ans) in enumerate(data):
            self.guess(img.flatten()/n)
            self.backProp(ans)
            if (idx+1)%self.batchSize == 0:
                self.update(self.batchSize/self.lr)

    def test(self, testIn, testAns):
        correct = 0
        cost = 0
        for img,ans in zip(testIn,testAns):
            if self.guess(img.flatten()/255) == ans:
                correct += 1
            cost +=self.cost(ans)
        return correct/len(testIn), cost/len(testIn)
    
    def testShow(self, testIn, testAns, maxima):
        correct = 0
        for img,ans in zip(testIn[:maxima],testAns):
            guess = self.guess(img.flatten()/255)
            if guess == ans: 
                correct += 1
            printImage(img)
            print("AI guessed", guess, "correct answer was", ans)
            input()
        return correct/len(testIn)
    
    def train(self, trainIn, trainAns, epochs):
        for i in range(epochs):
            self.resetDelta()
            t1 = time.time()
            self.runEpoch(trainIn, trainAns)
            print("Epoch ran in: ", round(time.time()-t1,4), " seconds")
            score = self.test(trainIn[:10000],trainAns[:10000])
            print("Score for the network after ", i+1, " epochs is ", score)
            