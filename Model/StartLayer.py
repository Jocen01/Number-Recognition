import numpy as np
import random as r
from .Sigmoid import sigmoid, sigmoidDer
class StaryLayer:
    def __init__(self, nbrNodes) -> None:
        self.nodes = np.zeros(nbrNodes)
