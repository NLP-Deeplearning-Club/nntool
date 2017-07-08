from abc import ABC, abstractmethod, abstractproperty

class FunctionABC(ABC):
    x = None
    y = None
    Theta = None
    djdTheta = None
    djdx = None
    djdy = None


    @abstractmethod
    def __call__(self,x):
        """计算模型的正向计算结果,并将其保存为self.z"""
        pass

    @abstractmethod
    def d_Theta(self,djdy):
        """对Theta的偏导"""
        pass

    @abstractmethod
    def d_x(self,djdy):
        """对Theta的偏导"""
        pass

    @abstractmethod
    def d(self,djdy):
        """对Theta的偏导"""
        pass
