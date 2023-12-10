import numpy as np
class MSE:
    def arr_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return max(0, arr.sum())

    def arr_function_derivative(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        return 1 if arr.sum() > 0 else 0