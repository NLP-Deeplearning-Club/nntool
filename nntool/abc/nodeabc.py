from abc import abstractmethod, abstractproperty
from .functionabc import FunctionABC

class NodeABC(FunctionABC):
    """__init__中初始化超参数"""

    @abstractproperty
    def z(self):
        """本步时候的计算结果"""

    @abstractmethod
    def __call__(self,In):
        """计算模型的正向计算结果,并将其保存为self.z"""
        pass
    @abstractmethod
    def d_Theta(self):
        """对Theta的偏导"""
        pass
