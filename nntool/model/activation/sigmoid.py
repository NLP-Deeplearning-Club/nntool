from ..modelabc import ModelABC
from math import exp

class Sigmoid(ModelABC):

    def __call__(self,In:float)->float:
        self.out = 1/(1+exp(-In))
        return self.out

    def backpropagation(self):
        pass
