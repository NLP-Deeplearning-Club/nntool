"""
感知器
"""
from nntool.abc.neuronabc import NodeABC

class LinearNeuron(NodeABC):

    def __init__(self,Theta):
        self.Theta = Theta

    def __call__(self,x):
        """计算模型的正向计算结果,并将其保存为self.z"""
        self.x = x
        self.y = self.Theta[:-1]*x+self.Theta[-1]
        return self.out

    def d_Theta(self):
        """对Theta的偏导"""
        djdb = self.djdy
        djdw = np.dot(self.x.reshape(-1,1),self.djdy.reshape(1,-1)).T
        return djdw,djdb

    def d_x(self):
        """对Theta的偏导"""
        djdx = np.dot(self.djdy,self.Theta[:-1].T)
        return djdx

    def d(self,djdy):
        self.djdy = djdy
        djdw,djdb = self.d_Theta()
        djdx = self.d_x()
        self.djdTheta = (djdw,djdb)
        self.djdx = djdx
        return self.djdx,self.djdTheta
