from nntool.abc.functionabc import FunctionABC
from math import exp

class Sigmoid(FunctionABC):

    def __call__(self,x):
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x=x
        self.y = 1/(1+exp(-self.x))
        return self.y


    def d_Theta(self):
        """对Theta的偏导"""
        pass


    def d_x(self):
        """对Theta的偏导"""
        djdx = self.djdy*(self.y(1-self.y))
        return djdx

    def d(self,djdy):
        self.djdy = djdy
        self.djdx = self.d_x()
        return self.djdx,None
