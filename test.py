
from keras.datasets import mnist
from Model.Layer import Layer
from Model.Ml import Ml
import numpy as np

#loading the dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(train_x.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_x.shape))
print('Y_test:  '  + str(test_y.shape))
ai = Ml([28**2,50,50,10])
from matplotlib import pyplot
def getComand():
    return int(input("""
train untill saturation: 1
run epochs: 2
show on test data: 3
quit : -1
: """))
comand = getComand()
while comand != -1:
    if comand == 1:
        ai.train(train_x, train_y,50)
        print("final score on test data is ", ai.test(test_x[:10000],test_y[:10000]))
    elif comand == 2:
        ai.train(train_x, train_y,int(input("How many epochs: ")))
        print("final score on test data is ", ai.test(test_x[:10000],test_y[:10000]))
    elif comand == 3:
        ai.test_show(test_x,test_y, int(input("How many examples: ")))
    comand = getComand()

