from nntool.abc.layerabc import NeuronLayer
from nntool.neuron import LinearNeuron
from nntool.utils.init_factory import uniform_factory
import numpy as np
class LinearNeuronLayer(NeuronLayer):
    """线性神经元层,也可以说是全连接层"""
    def forward(self,x:'array')->'array':
        self.x = x
        self.input_size = len(self.x)

        if not self._neurons:
            self.Thetas = np.array([self._init_factory(self.input_size+1) for i in range(self.size)])
            self._neurons = [LinearNeuron(i) for i in self.Thetas]
            self.shape = self.Thetas.shape

        result = []
        for i in self._neurons:
            result.append(i(x))
        self.y=np.array(result)
        return self.y

    def backward(self,djdys:'array',eta)->'djdTheta':
        self.djdys = djdys
        djdThetas = []
        djdxs = []
        for i,djdy in zip(self._neurons,djdys):

            djdx,djdTheta=i.d(djdy)

            djdThetas.append(djdTheta)
            djdxs.append(djdx)
        self.djdThetas = np.array(djdThetas)
        self.djdxs = np.array(djdxs)
        self.update_Theta(eta)

        return self.djdxs,eta


    def update_Theta(self,eta):
        """计算本层输出"""
        self.Thetas += -eta * self.djdThetas
        for i,new_Theta in zip(self._neurons,self.Thetas):
            i.update_Theta(new_Theta)



    def __init__(self,size:int,init_factory=uniform_factory()):
        self._size=size
        self._init_factory = init_factory

    @property
    def size(self):
        """本层的纬度"""
        return self._size
