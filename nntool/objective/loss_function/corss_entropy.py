from nntool.abc.functionabc import FunctionABC
import numpy as np

class CrossEntropy(FunctionABC):

    def __call__(self,hat_y,y):
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.y = y
        index = np.argwhere(y== True)
        self.loss = -np.log(hat_y[index])
        return self.loss

    def backward(self,eta)->'djdTheta':
        self.djdys = djdys
        self.djdxs = self.d(djdys)

        return self.y,eta

    def d(self,djdy,eta):
        pass
