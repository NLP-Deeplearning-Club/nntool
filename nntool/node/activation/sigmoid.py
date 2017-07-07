from nntool.abc.functionabc import FunctionABC
from math import exp

class Sigmoid(FunctionABC):

    def __call__(self,In:float)->float:
        self.out = 1/(1+exp(-In))
        return self.out

    def backpropagation(self):
        pass
