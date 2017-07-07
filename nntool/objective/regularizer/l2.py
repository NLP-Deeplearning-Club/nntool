from nntool.abc.functionabc import FunctionABC


class L2_Norm(FunctionABC):

    def __call__(Theta:'matrix')->float:
        n = Theta.shape[0]
        result = (self.lambd/(2*n))*((Theta**2).sum())
        return result
    def __init__(self,lambd:'>0'):
        """定义超参数"""
        self.lambd = lambd

    def dx(self):
        pass
