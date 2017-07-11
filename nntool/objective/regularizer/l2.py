from nntool.abc.functionabc import FunctionABC


class L2_Norm(FunctionABC):
    """l2正则项,"""
    def __call__(Theta):
        n = Theta.shape[0]
        result = (self.lambd/(2*n))*((Theta**2).sum())
        return result
    def __init__(self,lambd):
        """定义超参数"""
        self.lambd = lambd

    def dx(self):
        pass
