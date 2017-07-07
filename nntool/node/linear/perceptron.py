"""
感知器
"""
from nntool.abc.functionabc import FunctionABC

class Perceptron(FunctionABC):

    def __call__(self,In:'array')->float:
        self.out = self.W*In+self.b
        return self.out
    def __init__(self,W:'array',b:float):
        self.W = W
        self.b = b

    def dx(self):
        pass
