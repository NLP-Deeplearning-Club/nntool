from nntool.abc.layerabc import NeuronLayer
from nntool.utils.init_factory import uniform_factory
import numpy as np
class LinearNeuronLayer(NeuronLayer):
    """线性神经元层,也可以说是全连接层,概念上应该是这层中包含多个神经元,每个神经元单独计算,
    但本处的实现并没有使用神经元,而是使用参数矩阵"""
    def __init__(self,size,init_factory=uniform_factory()):
        self._size=size
        self._init_factory = init_factory
        self.Thetas = None

    @property
    def size(self):
        """本层的纬度"""
        return self._size

    def forward(self,x):
        self.x = x
        self.input_size = len(self.x)
        if self.Thetas is None:
            self.Thetas = self._init_factory(size=(self.size,self.input_size+1))
            self.shape = self.Thetas.shape
        W = self.Thetas[:,:-1]
        b = self.Thetas[:,-1]
        self.y = x.dot(W.T)+b
        return self.y

    def _d_Theta(self):
        """对Theta的偏导
        :math:`\\frac{\\partial J}{\\partial W} = \\frac{\\partial J}{\\partial y}`"""
        djdb = self.djdys
        djdwT = (self.x.reshape(-1, 1)*self.djdys.reshape(1, -1))
        return np.row_stack((djdwT, djdb)).T

    def _d_x(self):
        """对Theta的偏导:
        :math:`\\frac{\\partial J}{\\partial x} = \\frac{\\partial J}{\\partial y} W^T`
        """
        djdx = self.djdys.dot(self.Thetas[:,:-1])
        return djdx

    def d(self, djdys):
        self.djdys = djdys
        self.djdThetas = self._d_Theta()
        self.djdxs = self._d_x()
        return self.djdxs, self.djdThetas

    def backward(self,djdys,eta):
        self.djdys = djdys
        self.djdxs, self.djdThetas = self.d(djdys)
        self.update_Theta(eta)

        return self.djdxs,eta


    def update_Theta(self,eta):
        """计算本层输出"""
        self.Thetas += -eta * self.djdThetas
