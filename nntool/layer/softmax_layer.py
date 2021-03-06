from nntool.abc.layerabc import ActivationFunctionLayer

import numpy as np
class SoftmaxLayer(ActivationFunctionLayer):
    """softmax函数用于将输入的向量转换为每项值域在[0.1]且各项相加和为1的与输入向量等长的向量.
    这个向量可以作为多值分类的各位对应的值的可能性."""

    def forward(self,x):
        """计算模型的正向计算结果,并将其保存为
        :math:`softmax(x)_j=\\frac {e^xj} {\\sum_{i=1}^d e^{x_i}}`
        """
        self.x = x
        self.input_size = len(self.x)
        self._size = self.input_size

        ex = np.exp(x)
        ex = ex / np.sum(ex)
        self.y = ex
        return self.y
    @property
    def size(self):
        """本层的输出纬度"""
        return self._size

    def backward(self,djdys,eta):
        self.djdys = djdys
        self.djdxs = self.d(djdys)
        return self.djdxs,eta


    def d(self,y):
        """
        :math:`\\frac{\\partial y_i}{\\partial x_i} = y_i(1-y_i);\\frac{\\partial y_i}{\\partial x_j} = - y_iy_j`
        """
        index = np.argwhere(y== True)
        djdx = self.y

        djdx[index] -= 1
        return djdx
