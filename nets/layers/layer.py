from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    '''
    定义所有层的基类，新定义的层必须重写方法:正向传播和反向传播
    Define the base class for all layers, and newly defined layers must rewrite methods: forward and backward progress function
    '''

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        # 输入为损失关于a_i 的梯度
        # 计算本层中正向过程输出关于出入的梯度 diff
        # 输出损失关于a_i-1的梯度
        pass