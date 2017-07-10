from nntool.abc.layerabc import LayerABC
class InputLayer(LayerABC):
    def forward(self,z)->'array':
        self.y = self.x
        return self.y

    def __init__(self,x:'array'):
        self.x = x
        self._size=len(x)
    @property
    def size(self)->'array':
        """本层的纬度"""
        return self._size
