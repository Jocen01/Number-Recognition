import numpy as np

from .Sigmoid import sigmoid, sigmoidDer
class Layer:
    def __init__(self, nbr_nodes, prev_nbr_nodes, prev_layer = None, func = sigmoid, func_derivative = sigmoidDer) -> None:
        self.nbr_nodes = nbr_nodes
        self.prev_layer = prev_layer
        self.nodes = np.zeros((nbr_nodes,1))
        self.func = np.vectorize(func)
        self.func_derivative = np.vectorize(func_derivative)
        self.weights = np.array([[np.random.normal() for _ in range(prev_nbr_nodes)] for _ in range(self.nbr_nodes)])
        self.bias = np.array([[np.random.normal()]for _ in range(self.nbr_nodes)])
        self.delta_weights = np.zeros(self.weights.shape)
        self.delta_bias = np.zeros(self.bias.shape)
        self.layer_nbr = -1

    def mul(self, vec):
        self.nodes = self.weights @ vec.reshape(-1,1) + self.bias
        return self.func(self.nodes)
    
    def backpropagation(self, dCdA):
        dZ = self.func_derivative(self.nodes)
        dN = (dZ * dCdA)
        self.backpropagation_weights(dN)
        self.backpropagation_bias(dN)
        return self.backpropagation_calc_dCdAj(dN)

    def backpropagation_weights(self, dN):
        dW = dN @ self.prev_layer.nodes.T
        self.delta_weights += dW

    def backpropagation_bias(self, dN):
        self.delta_bias += dN

    def backpropagation_calc_dCdAj(self, dN):
        nextdCdA = self.weights.T @ dN
        return nextdCdA

    def MSE(self, correctArray):
        dCdAj = (self.func(self.nodes) - correctArray) * 2
        return dCdAj
    
    def update_layer(self, lr):
        self.weights -= self.delta_weights*lr
        self.bias -= self.delta_bias*lr
        self.reset_delta()

    def reset_delta(self):
        self.delta_weights = np.zeros(self.weights.shape)
        self.delta_bias = np.zeros(self.bias.shape)

