import numpy as np

from ..parameter import Parameter
from .lr_scheduler import LRScheduler

'''
L2 正则化：在代价函数后加2正则化项 '\frac{\lambda}{2}||W||_{2}^{2}'
正则化项是独立的分量，在加入之后并没有改变原来部分的公式
每次训练都会按一定的比例减小参数值，所以L2正则化也叫权值衰减
W = (1-\lambda)W
'''

class SGD(object):
    def __init__(self, parameters, LRsche, decay=0) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lrSche = LRsche
        self.lr = LRsche.lr
        self.decay_rate = 1.0 - decay
    
    def update(self,loop):

        newlr = self.lrSche.lrUpdate(loop)
        # if newlr != self.lr:
        #     # 学习率更新：
        #     print(np.linalg.norm(self.parameters[0].data))
        self.lr = newlr

        # 权值衰减
        for p in self.parameters:
            if self.decay_rate < 1: p.data *= self.decay_rate
            p.data -= self.lr * p.grad
    

