from nntool.abc.layerabc import ActivationFunctionLayer

import numpy as np
class TanhLayer(ActivationFunctionLayer):

    def forward(self,x:'array')->'array':
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x = x
        self.input_size = len(self.x)

        self.y = np.tanh(x)
        return self.y


    def backward(self,djdys:'array',eta)->'djdTheta':
        self.djdys = djdys
        self.djdxs = self.d_x()

        return self.djdxs,eta


    def d_x(self):
        """对Theta的偏导"""
        djdx = self.djdys*(self.y(1-self.y))
        return djdx

    def d(self,y):
        pass
