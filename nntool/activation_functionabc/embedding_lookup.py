from nntool.abc.functionabc import FunctionABC
import numpy as np
class EmbeddingLookup(FunctionABC):

    def __call__(self,x):
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x = x
        self.y = self.Theta[self.x]
        return self.y

    def __init__(self,Theta):
        """theta 必须是一个d*v维的矩阵,d为数据的输出纬度,v自己定,Theta是超参数"""
        self.Theta = Theta

    def d_Theta(self):
        """对Theta的偏导"""
        djdw = np.zeros_like(self.Theta)
        djdw[self.x] = self.djdy
        return djdw




    def d_x(self):
        """对Theta的偏导"""
        pass

    def d(self,djdy):
        self.djdy = djdy
        self.djdTheta= self.d_Theta()
        return None,self.djdTheta
