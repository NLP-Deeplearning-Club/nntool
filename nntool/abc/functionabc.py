from abc import ABC, abstractmethod, abstractproperty

class FunctionABC(ABC):
    """__init__中初始化超参数和"""

    @abstractmethod
    def __call__(self,In):
        """计算模型的正向计算结果,并将其保存为self.z"""
        pass


    @abstractmethod
    def d_Theta(self):
        """对Theta的偏导"""
        pass
