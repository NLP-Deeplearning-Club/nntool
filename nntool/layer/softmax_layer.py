from nntool.abc.layerabc import ActivationFunctionLayer

import numpy as np
class SoftmaxLayer(ActivationFunctionLayer):

    def forward(self,x:'array')->'array':
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x = x
        self.input_size = len(self.x)

        ex = np.exp(x)
        ex = ex / np.sum(ex)
        self.y = ex
        return self.y


    def backward(self,djdys:'array',eta)->'djdTheta':
        self.djdys = djdys
        self.djdxs = self.d(djdys)

        return self.djdxs,eta


    def d(self,y):
        index = np.argwhere(y== True)
        djdx = self.y

        djdx[index] -= 1
        return djdx
