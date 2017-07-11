from nntool.abc.layerabc import ActivationFunctionLayer

import numpy as np
class TanhLayer(ActivationFunctionLayer):
    """双曲正弦函数,值域在[-1,1]的激活函数,形状与sigmoid函数类似,一般用于做挤压
    """
    def forward(self,x):
        """计算模型的正向计算结果
        :math:`\\tanh(x)=\\frac {1-e^{-2x}} {1+e^{-2x}}`"""
        self.x = x
        self.input_size = len(self.x)

        self.y = np.tanh(x)
        return self.y


    def backward(self,djdys,eta):
        self.djdys = djdys
        self.djdxs = self.d_x()

        return self.djdxs,eta


    def d_x(self):
        """对x的偏导
        :math:`\\frac{\\partial y}{\\partial x} = 1-y^2`
        """
        djdx = self.djdys*(1-self.y**2)
        return djdx

    def d(self,y):
        pass
