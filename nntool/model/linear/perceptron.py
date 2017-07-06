"""
感知器
"""
from ..modelabc import ModelABC

class Perceptron(ModelABC):

    def __call__(self,In:'array')->float:
        self.out = self.W*In+self.b
        return self.out
    def __init__(self,W:'array',b:float):
        self.W = W
        self.b = b

    def backpropagation(self):
        pass
