import numpy as np
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidDer(x):
    s = (1/(1 + np.exp(-x)))
    return s*(1-s)