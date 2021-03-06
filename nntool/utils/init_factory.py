"""
初始化工具
"""
import numpy as np

def normal_factory(range_=(1,0.5)):
    def  _warp(size):
        return np.random.normal(*range_,size=size)
    return _warp

def uniform_factory(range_=(1,0.5)):
    def  _warp(size):
        return np.random.uniform(*range_,size=size)
    return _warp
