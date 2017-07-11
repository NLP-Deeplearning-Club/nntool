from abc import ABC, abstractmethod, abstractproperty
from nntool.abc.functionabc import FunctionABC


class LayerABC(ABC):
    """层这是一个抽象概念,当多个结果要进行相同或者相关操作的时候,这就可以被看作是一层.
    层也有特殊的地方,比如带有超参数的的就要设定超参数,有的有参数的就需要每次训练更新参数.
    """
    x = None
    y = None
    djdys = None

    @abstractmethod
    def forward(self, x):
        """计算本层正向输出,"""
    @abstractmethod
    def backward(self, x):
        """计算本层输出"""
        pass




class HiddenLayer(LayerABC):
    """隐藏层是指除去输入输出的所有层"""
    @abstractproperty
    def size(self):
        """本层的输出纬度"""


class NeuronLayer(HiddenLayer):
    """神经元层,表示本层都是神经元,
    输入的每一个纬度都会进入层内的每个神经元进行计算,神经元一般都有参数,而训练这些参数也就是我们的任务"""
    _neurons = None
    djdThetas = None
    djdxs = None
    input_size = None
    size = None
    Thetas = None
    @abstractmethod
    def update_Theta(self, eta, djdTheta):
        """更新参数"""
        pass



class ActivationFunctionLayer(HiddenLayer):
    """激活函数层一般只是将上一层传入的输入进行非线性变换,再输出其结果"""
    pass
