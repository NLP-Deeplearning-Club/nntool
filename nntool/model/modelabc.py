from abc import ABC, abstractmethod

class ModelABC(ABC):
    """__init__中初始化超参数"""
    @abstractmethod
    def __call__(self,In):
        """计算模型的正向计算结果,并将其保存为self.y"""
        pass
    @abstractmethod
    def backpropagation(self):
        """反向传播算法的结果"""
        pass
