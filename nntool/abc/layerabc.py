from abc import ABC, abstractmethod, abstractproperty
from nntool.abc.functionabc import FunctionABC


class LayerABC(ABC):
    """__init__中初始化超参数,y为本层的输出"""
    x = None
    y = None
    djdys = None

    @abstractmethod
    def forward(self, x):
        """计算本层输出"""




class HiddenLayer(LayerABC):

    @abstractmethod
    def backward(self, x):
        """计算本层输出"""
        pass

    @abstractproperty
    def size(self):
        """本层的输出纬度"""


class NeuronLayer(HiddenLayer):
    _neurons = None
    djdThetas = None
    djdxs = None
    input_size = None
    size = None
    Thetas = None
    @abstractmethod
    def update_Theta(self, eta, djdTheta):
        """计算本层输出"""
        pass



class ActivationFunctionLayer(HiddenLayer, FunctionABC):
    pass
