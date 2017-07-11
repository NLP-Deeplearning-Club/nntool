from nntool.abc.layerabc import ActivationFunctionLayer

import numpy as np
class SigmoidLayer(ActivationFunctionLayer):
    """logistic激活函数层,作用是将线性层传入的内容映射到,
    [0,1]区间,其两端平缓中间陡峭,因此在两端容易出现梯度消失的情况,
    Sigmoid的用途一般可以归结为:
    做概率
    做开关
    做"挤压"

    """
    def forward(self,x):
        """计算模型的正向计算结果
        :math:`\\sigma(x)=\\frac 1 {1+e^{-x}}`
        """
        self.x = x
        self.input_size = len(self.x)

        self.y = 1/(1+np.exp(-self.x))
        return self.y



    def backward(self,djdys,eta):
        self.djdys = djdys
        self.djdxs = self.d_x()
        return self.djdxs,eta


    def d_x(self):
        """对x的偏导
        :math:`\\frac{\\partial y}{\\partial x} = y(1-y)`
        """
        djdx = self.djdys*(self.y(1-self.y))
        return djdx
