from nntool.abc.functionabc import FunctionABC
def (y:'array',out:'array'):
    """均方差损失函数,常用于nnl"""
    n= len(y)
    return square_loss(y,out)/n
class MSE(FunctionABC):

    def __call__(self,In:'array')->float:
        self.out = self.W*In+self.b
        return self.out
    def __init__(self,W:'array',b:float):
        self.W = W
        self.b = b

    def dx(self):
        pass
