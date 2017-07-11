from abc import abstractmethod, abstractproperty
from .functionabc import FunctionABC

class NeuronABC(FunctionABC):
    """神经元其实也是一个函数,神经网络算法本质上就是训练各个神经元的参数"""
    pass
