import random
import numpy as np
class BGD:

    def forward(self,x,i=0):
        print("forward {i} layer".format(i=i))
        if i == self.layers_len:
            print('result:{re}'.format(re=x))
            return x
        else:
            y = self.layers[i].forward(x)
            i += 1
            print('result:{re}'.format(re=y))
            return self.forward(y,i)

    def backward(self,y,eta,i = None):
        if i:
            print("backward {i} layer".format(i=i))
            if i == 0:
                result,eta = self.layers[i].backward(y,eta)
                print(result)
                return True
            else :
                result,eta = self.layers[i].backward(y,eta)
                print(result)
                i -= 1
                return self.backward(result,eta,i)
        else:
            print("backward objective function")
            result,eta = self.objective.backward(eta)
            print(result)
            return self.backward(result,eta,i=self.layers_len-1)


    def __call__(self,model):
        self.layers = model._layers
        self.layers_len = len(self.layers)

        for i in range(self.epoch):
            print("epoch {i}==============".format(i=i))
            batch = random.sample(list(zip(self.X,self.Y)),self.batch_size)
            for (x,y) in batch:
                result = self.objective(self.forward(x),y)
                self.backward(y,self.eta)
            print("loss:{i}".format(i = result))
            print("epoch {i} end==============".format(i=i))



    def __init__(self,X,Y,objective,batch_size,eta=0.1,epoch = 10):
        self.X = X
        self.Y = Y
        self.objective
        self.batch_size = batch_size
        self.eta = eta
        self.epoch = epoch
