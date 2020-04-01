import torch


class CubemapTensor():

    def __init__(self, sides):
        torch.Tensor
        self.sides = sides

    def clone(self, *args, **kwargs):
        return CubemapTensor({s: x.clone(*args, **kwargs) for s, x in self.sides.items()})

    def to(self, *args, **kwargs):
        return CubemapTensor({s: x.to(*args, **kwargs) for s, x in self.sides.items()})

    def __getattr__(self, item):
        return
