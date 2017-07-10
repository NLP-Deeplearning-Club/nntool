from abc import ABC, abstractmethod, abstractproperty

class FunctionABC(ABC):

    @abstractmethod
    def __call__(self,x):
        """计算模型的正向计算结果,并将其保存为self.z"""
        pass



    @abstractmethod
    def d(self,djdy):
        """对Theta的偏导"""
        pass
