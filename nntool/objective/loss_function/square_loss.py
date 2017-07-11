from nntool.abc.functionabc import FunctionABC
def square_loss(y,out):
    """平方损失函数"""
    return ((y-out)**2).sum()
