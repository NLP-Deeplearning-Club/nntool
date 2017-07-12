from nntool.abc.modelabc import ModelABC
import numpy as np
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

    def fit(self,dev_x,dev_y):
        """测试在dev数据集上的效果"""
        total = 0
        correct = 0
        for i in range(len(dev_y)):
            total += 1
            if self.predict(dev_x[i]).argmax() == dev_y[i].argmax():
                correct += 1
        correct_rate = correct/total
        print('total:{total},corrcet:{correct},Correct rate:{correct_rate}'.format(
            total=total,correct=correct,correct_rate=correct_rate))

    def _forward(self,x,i=0):
        """前向运算"""
        #print("forward {i} layer".format(i=i))
        if i == len(self._layers):
            #print('result:{re}'.format(re=x))
            return x
        else:
            y = self._layers[i].forward(x)
            i += 1
            #print('result:{re}'.format(re=y))
            return self._forward(y,i)

    def predict_probability(self,x_test):
        result = self._forward(x_test)
        return result

    def predict(self,x_test):
        """预测数据"""
        probabilitys = self.predict_probability(x_test)
        maxindex = probabilitys.argmax()
        result = np.array([True if i == maxindex else False for i in range(
            len(probabilitys))])
        return result
