from nntool.abc.modelabc import ModelABC

class Sequential(ModelABC):

    """序列模型,这个是keras中的概念,将模型理解为层的堆叠
    """
    _layers = []
    _trained = False

    def add(self,layer:'layer'):
        self._layers.append(layer)

    @property
    def trained(self):
        """是否已经训练过"""
        return self._trained

    def train(self,trainner):
        """训练模型"""
        trainner(self)
        self._trained = True

    def fit(self,dev:'matrix'):
        """测试在dev数据集上的效果"""
        if not self.trained:
            raise AttributeError("train the model first")
        else:
            pass
    def predict(x_test):
        """预测一组数据"""
        pass
