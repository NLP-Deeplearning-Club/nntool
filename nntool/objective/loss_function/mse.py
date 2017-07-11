from nntool.abc.functionabc import FunctionABC

class MSE(FunctionABC):
    """均方差损失函数,常用于nnl"""
    def __call__(self,hat_y,y):

        return ((hat_y-y)**2).sum()


    def dx(self):
        pass
