import numpy as np
def sigmoidDer(x):
        s = (1/(1 + np.exp(-x)))
        return s*(1-s)

class Softmax:
    f = np.vectorize(lambda x: 1/(1 + np.exp(-x)))
    der = np.vectorize(sigmoidDer)
    
    def arr_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return Softmax.f(arr)

    def arr_function_derivative(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return Softmax.der(arr)

    