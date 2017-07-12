from nntool.abc.neuronabc import NeuronABC
import numpy as np


class LinearNeuron(NeuronABC):
    """
    线性神经元,也就是感知器,也就是W*X+b,w为对应输入的权重,b为偏置.
    """

    def __init__(self, Theta):
        self.Theta = Theta

    def __call__(self, x):
        """计算模型的正向计算结果,并将其保存为self.z

        :math:`y = Wx + b`
        """
        self.x = x
        self.y = (self.Theta[:-1] * x).sum() + self.Theta[-1]
        return self.y

    def d_Theta(self):
        """对Theta的偏导
        :math:`\\frac{\\partial J}{\\partial W} = \\frac{\\partial J}{\\partial y}`"""
        djdb = np.array([self.djdy])
        djdw = np.dot(self.x.reshape(-1, 1), self.djdy.reshape(1, -1)).T[0]
        return np.hstack((djdw, djdb))

    def d_x(self):
        """对Theta的偏导:
        :math:`\\frac{\\partial J}{\\partial x} = \\frac{\\partial J}{\\partial y} W^T`
        """
        djdx = np.dot(self.djdy, self.Theta[:-1].T)
        return djdx

    def d(self, djdy):
        self.djdy = djdy
        self.djdTheta = self.d_Theta()
        self.djdx = self.d_x()
        return self.djdx, self.djdTheta

    def update_Theta(self,new_Theta):
        self.Theta = new_Theta
