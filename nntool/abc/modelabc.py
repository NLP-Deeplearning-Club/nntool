from abc import ABC, abstractmethod, abstractproperty

class ModelABC(ABC):
    """模型是由层组合而成的运算结构,神经网络中模型基本上就是花式堆叠层."""
    @abstractmethod
    def add(self,layer):
        """添加一层"""
    @abstractproperty
    def trained(self):
        """是否已经训练过"""
    @abstractmethod
    def train(self):
        """计算模型的正向计算结果,并将其保存为self.y"""
        pass

    @abstractmethod
    def fit(self,dev):
        """反向传播算法的结果"""
        pass
    @abstractmethod
    def predict(x_test):
        """预测一组数据"""
