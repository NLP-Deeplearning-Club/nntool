from nntool.abc.functionabc import FunctionABC

class ReLu(FunctionABC):

    def __call__(self,x):
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x = x
        self.y = max(0,self.x)
        return self.y


    def d_Theta(self):
        """对Theta的偏导"""
        pass



    def d_x(self):
        """对Theta的偏导"""
        dydx = 1 if self.x> 0 else 0
        djdx = self.djdy*dydx
        return djdx

    def d(self,djdy):
        self.djdy = djdy
        self.djdx = self.d_x()
        return self.djdx,None
