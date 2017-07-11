from nntool.abc.functionabc import FunctionABC
def hinge(y,out):
    """常用于最大化利润,常用在svm,需要与其输出为+-1"""
    if all(out**2-1):
        return max(0,1-y*out)
    else:
        raise AttributeError("out must be +-1")
