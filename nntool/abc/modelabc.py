from abc import ABC, abstractmethod, abstractproperty

class ModelABC(ABC):
    """__init__中初始化超参数"""
    @abstractmethod
    def add(self,layer:'layer'):
        """添加一层"""
    @abstractproperty
    def trained(self):
        """是否已经训练过"""
    @abstractmethod
    def train(self):
        """计算模型的正向计算结果,并将其保存为self.y"""
        pass

    @abstractmethod
    def fit(self,dev:'matrix'):
        """反向传播算法的结果"""
        pass
    @abstractmethod
    def predict(x_test):
        """预测一组数据"""
