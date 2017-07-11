from nntool.abc.functionabc import FunctionABC
import numpy as np

class CrossEntropy(FunctionABC):
    """交叉熵损失函数,一般配合softmax一起使用"""
    def __call__(self,hat_y,y):
        """$ J = CE(y,j) $"""
        self.y = y
        index = np.argwhere(y== True)
        self.loss = -np.log(hat_y[index])
        return self.loss

    def backward(self,eta):
        """交叉熵现在只和softmaxt配合使用"""
        return self.y,eta

    def d(self,djdy,eta):
        pass
