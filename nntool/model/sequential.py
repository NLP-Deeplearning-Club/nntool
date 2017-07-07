from nntool.abc.modelabc import ModelABC

class Sequential(ModelABC):
    _layers = []
    _trained = False


    def forward_propagation(self,z=None,n=0):
        if n == 0:
            new_z = self._layers[0]()
        else:
            new_z = self._layers[n]()

        if n == len(self._layers):
            return new_z
        else:
            new_n = n+1
            return forward_propagation(new_z,new_n)


    def add(self,layer:'layer'):
        self._layers.append(layer)

    @property
    def trained(self):
        """是否已经训练过"""
        return self._trained

    def train(self,tranner):
        """计算模型的正向计算结果,并将其保存为self.y"""
        tranner(self)
        self._trained = True

    def fit(self,dev:'matrix'):
        """反向传播算法的结果"""
        if not self.trained:
            raise AttributeError("train the model first")
        else:
            [ for i in self._layers]
    @abstractmethod
    def predict(x_test):
        """预测一组数据"""
