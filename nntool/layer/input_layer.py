from nntool.abc.layerabc import LayerABC
class InputLayer(LayerABC):
    _z=None

    def __call__(self):
        self._z = self.x
        return self._z

    def __init__(self,x:'array'):
        self.x = x
        self._size=len(x)
    @property
    def size(self):
        """本层的纬度"""
        return self._size
    @property
    def z(self):
        """本曾的输出"""
        return self._z
