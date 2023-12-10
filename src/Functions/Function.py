from typing import Protocol
import numpy as np

class Function(Protocol):
    def arr_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        ...

    def arr_function_derivative(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        ...

class CostFunction(Protocol):
    def cost(inp: np.ndarray[np.float64], target: np.ndarray[np.float64]) -> np.float64:
        ...

    def cost_derivative(inp: np.ndarray[np.float64], target: np.ndarray[np.float64]) -> np.float64:
        ...