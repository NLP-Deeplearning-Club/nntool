from enum import Enum
from collections.abc import Sequence
import numpy as np
class OneHot:
    def __init__(self,name,seq):
        seq_set = list(set(seq))
        self.size = len(seq_set)
        self._enum = Enum(name, seq_set,  start=0)

    def _encode(self,obj):
        en = self._enum.__members__.get(obj,None)
        if en is None:
            raise AttributeError("unknow attribute:{}".format(obj))
        else:
            return np.array([True if i == en.value else False for i in range(self.size)])
    def encode(self,obj):
        if isinstance(obj,np.ndarray) or (isinstance(obj,Sequence) and not isinstance(obj,str)):
            result = np.array([self._encode(i) for i in obj])
        else:
            result = self._encode(obj)
        return result

    def _decode(self,onehot):
        if len(onehot) != self.size:
            raise AttributeError("attribute {}'s size is not match,size must be {}".format(
                onehot,self.size))
        onehot_index = np.argwhere(onehot == True)
        if onehot_index.shape != (1,1):
            raise AttributeError("attribute {} is not a one-hot vector".format(onehot))

        temp = [i.name for i in self._enum if i.value==onehot_index[0][0]][0]
        return temp

    def decode(self,onehot):
        if len(onehot.shape)==1:
            result = self._decode(onehot)
        else:
            result = [self._decode(i) for i in onehot]
        return result
