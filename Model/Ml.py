from .Layer import Layer
from .StartLayer import StaryLayer
from .Utilities import *
import numpy as np
import random
import time
class Ml:
    def __init__(self, layers, batch_size=100, learn_rate=0.01, cost_func=None) -> None:
        self.layers = [StaryLayer(layers[0])] + [Layer(i,layers[idx]) for idx ,i in enumerate(layers[1:])]
        for i in range(len(self.layers)-1):
            self.layers[i+1].prev_layer = self.layers[i]
        self.start = self.layers[0]
        self.batch_size = batch_size
        self.lr = learn_rate
        self.const_function = np.vectorize(lambda val: val**2) if not cost_func else cost_func
        self.result = None

    def guess(self, arr):
        arr = arr.reshape(-1,1)
        self.start.nodes = arr
        for l in self.layers[1:]:
            arr = l.mul(arr)
        self.result = arr
        return np.argmax(arr)
    
    def cost(self, correct):
        c = np.zeros((self.layers[-1].nbr_nodes,1))
        c[correct,0] = 1
        return sum(self.const_function(self.result-c))[0]
    
    def backpropagation(self, correct):
        correct_arr = np.zeros((self.layers[-1].nbr_nodes,1))
        correct_arr[correct,0] = 1
        dCdA = self.layers[-1].MSE(correct_arr)
        for lay in self.layers[:0:-1]:
            dCdA = lay.backpropagation(dCdA)

    def reset_delta(self):
        for lay in self.layers[1:]:
            lay.reset_delta()
    
    def update(self):
        for lay in self.layers[1:]:
            lay.update_layer(self.lr)

    def run_epoch(self, train_x, train_y):
        n = np.max(train_x)
        data = list(zip(train_x, train_y))
        random.shuffle(data)
        for idx,(img,ans) in enumerate(data):
            self.guess(img.flatten()/n)
            self.backpropagation(ans)
            if (idx+1)%self.batch_size == 0:
                self.update()

    def test(self, test_x, test_y):
        correct = 0
        cost = 0
        for img,ans in zip(test_x,test_y):
            if self.guess(img.flatten()/255) == ans:
                correct += 1
            cost +=self.cost(ans)
        return correct/len(test_x), cost/len(test_x)
    
    def test_show(self, test_x, test_y, nbr):
        for img,ans in zip(test_x[:nbr],test_y):
            guess = self.guess(img.flatten()/255)
            printImage(img)
            print("AI guessed", guess, "correct answer was", ans)
            input()
    
    def train(self, trainIn, trainAns, epochs):
        for i in range(epochs):
            self.reset_delta()
            t1 = time.time()
            self.run_epoch(trainIn, trainAns)
            print("Epoch ran in: ", round(time.time()-t1,4), " seconds")
            score = self.test(trainIn[:10000],trainAns[:10000])
            print("Score for the network after ", i+1, " epochs is ", round(score,5))
            