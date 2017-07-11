from abc import ABC, abstractmethod, abstractproperty

class FunctionABC(ABC):
    """神经网络中的函数多是进行矩阵计算,需要可以正向求出值,也需要可以逆向求出对x和对参数Theta的偏导
    """

    @abstractmethod
    def __call__(self,x):
        """计算模型的正向计算结果"""
        pass



    @abstractmethod
    def d(self,djdy):
        """对Theta和对x的偏导,有的可能没有其中某一项但这不重要"""
        pass
class ActivationFunctionABC(FunctionABC):
    """激活函数一般是用来加入非线性因素的，因为线性模型的表达能力不够,一般接在线性层后面使用"""

class NormABC(FunctionABC):
    '''正则项函数,用于防止过拟合,正则化参数等价于对参数引入先验分布'''
    pass

class LossFuction(FunctionABC):
    """损失函数,用于直观评价模型预计值与真实值间的差距,用于反向传播计算神经元参数"""
    pass
