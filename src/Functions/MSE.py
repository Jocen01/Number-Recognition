import numpy as np
class MSE:
    def cost(inp: np.ndarray[np.float64], target: np.ndarray[np.float64]) -> np.float64:
        return np.array([(i-t)**2 for i,t in zip(inp,target)]).sum()

    def cost_derivative(inp: np.ndarray[np.float64], target: np.ndarray[np.float64]) -> np.float64:
        return (inp - target)*2