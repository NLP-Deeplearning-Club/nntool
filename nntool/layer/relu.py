from nntool.abc.layerabc import ActivationFunctionLayer

import numpy as np
class ReLULayer(ActivationFunctionLayer):
    """调整线性激活函数,这个函数的值域为[0,+oo],它是一个分段函数,两段都是线性的,
    这个函数来自于仿生学,可以解决sigmoid函数的梯度消失问题.

    标准的sigmoid输出不具备稀疏性,需要用一些惩罚因子来训练出一大堆接近0的冗余数据来,从而产生稀疏数据,
    例如L1、L1/L2或Student-t作惩罚因子.因此需要进行无监督的预训练.
    而ReLU是线性修正,是purelin的折线版.它的作用是如果计算出的值小于0,就让它等于0;否则保持原来的值不变.
    这是一种简单粗暴地强制某些数据为0的方法.然而经实践证明,训练后的网络完全具备适度的稀疏性.
    而且训练后的可视化效果和传统方式预训练出的效果很相似,这也说明了ReLU具备引导适度稀疏的能力.
    """
    def forward(self,x):
        """计算模型的正向计算结果
        :math:`ReLU(x)=max(0,x)`
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
        :math:`\\frac{\\partial y}{\\partial x} =1 if x>0; 0 if x<0`
        """
        djdx = self.djdys*(self.y(1-self.y))
        return djdx
