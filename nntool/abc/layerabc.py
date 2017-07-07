from abc import ABC, abstractmethod, abstractproperty

class LayerABC(ABC):
    """__init__中初始化超参数"""

    @abstractmethod
    def __call__():
        """计算本层输出"""

    @abstractproperty
    def size(self):
        """本层的纬度"""

    @abstractproperty
    def z(self):
        """本曾的输出"""
