
from keras.datasets import mnist
from Model.Layer import Layer
from Model.Ml import Ml
import numpy as np

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
ai = Ml([28**2,16,16,10])
from matplotlib import pyplot
for i in range(9):  
    g = ai.guess(np.ndarray.flatten(train_X[i]))
    print(g)
    pyplot.subplot(330 + 1 + i)
    print(type(train_X[i]))
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
tot = 0
correct = 0
for i in range(1000):
    g = ai.guess(np.ndarray.flatten(train_X[i]))
    print(g)
    tot += 1
    if g == train_y[i]:
        correct += 1
    print(correct/tot)
pyplot.show()
