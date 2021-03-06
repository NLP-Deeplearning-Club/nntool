from nntool.abc.layerabc import HiddenLayer
from nntool.utils.init_factory import uniform_factory


class EmbeddingLayer(HiddenLayer):
    """嵌入层,这层一般来说是用作将稀疏的输入矩阵(onehot编码)转为稠密矩阵,同时降低维度
    """

    x = None
    y = None
    djdys = None
    Theta = None

    def __call__(self,x):
        """计算模型的正向计算结果"""
        return self.forward(x)

    def __init__(self,Theta = None,*,size,init_factory=uniform_factory()):
        """theta 必须是一个d*v维的矩阵,d为数据的输出纬度,v自己定,Theta是超参数"""

        if Theta:
            self.Theta = Theta
            self._size = len(self.shape[0])
            self.shape = self.Theta.shape
        else:
            self._init_factory = init_factory
            self._size = size



    @property
    def size(self):
        """本层的输出纬度"""
        return self._size

    def d_Theta(self):
        """对Theta的偏导
        :math:`\\frac{\\partial y}{\\partial W[x,:]} = \\frac{\\partial J}{\\partial y}`
        """
        djdw = np.zeros_like(self.Theta)
        for index,djdy in zip(self.indexs,self.djdys):
            djdw[index] = self.djdy
        return djdw

    def d_x(self):
        """对x的偏导"""
        return NotImplemented

    def d(self,djdy):
        self.djdys= djdys
        self.djdTheta = self.d_Theta()
        return None,self.djdTheta

    def forward(self, x):
        """计算本层输出
        :math:`y=ebl(x)=W[x,:] w  \\in R^{|v|*d}`
        """
        self.x = x
        self.input_size = len(x)
        one_hot_len = len(self.x[0])
        if self.Theta:
            if one_hot_len != self.shape[0]:
                raise AttributeError("embeding must have the same input_size with the matrix's row")
        else:
            self.Theta = [self._init_factory(self.input_size) for i in range(self.size)]
            self.shape = self.Theta.shape
        self.indexs = np.array([np.argwhere(i== True)[0][0] for i in self.x])
        self.y = np.array([self.Theta[index] for index in self.indexs])
        return self.y

    def backward(self, djdy,eta):
        """计算本层输出"""
        _,djdTheta = self.d()
        self.djdTheta = djdTheta
        return self.djdTheta,eta

    def _update_Thetas(self,eta):
        self.Theta += -eta * self.djdTheta
