import numpy as np
class Softmax:
    def arr_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        s = arr.sum()
        return np.array(list(map(lambda x: x/s, arr))).reshape(shape=arr.shape)

    def arr_function_derivative(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        s = arr.sum()
        return np.array(list(map(lambda x: (s-x)/(s**2), arr))).reshape(shape=arr.shape)
