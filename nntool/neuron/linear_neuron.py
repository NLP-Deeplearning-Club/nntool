from nntool.abc.neuronabc import NeuronABC
import numpy as np


class LinearNeuron(NeuronABC):
    """
    线性神经元,也就是感知器,也就是W*X+b,w为对应输入的权重,b为偏置.
    其
    """

    def __init__(self, Theta: 'array'):
        self.Theta = Theta

    def __call__(self, x: 'array')->float:
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x = x
        self.y = (self.Theta[:-1] * x).sum() + self.Theta[-1]
        return self.y

    def d_Theta(self):
        """对Theta的偏导"""
        djdb = np.array([self.djdy])
        djdw = np.dot(self.x.reshape(-1, 1), self.djdy.reshape(1, -1)).T[0]
        return np.hstack((djdw, djdb))

    def d_x(self):
        """对Theta的偏导"""
        djdx = np.dot(self.djdy, self.Theta[:-1].T)
        return djdx

    def d(self, djdy: 'array')->'djdx,djdTheta':
        self.djdy = djdy
        self.djdTheta = self.d_Theta()
        self.djdx = self.d_x()
        return self.djdx, self.djdTheta

    def update_Theta(self,new_Theta):
        self.Theta = new_Theta
