import numpy as np

from ..parameter import Parameter
from .layer import Layer


class Linear(Layer):
    def __init__(self,shape,requires_grad=True,bias=True,**kwargs) -> None:
        '''
        shape:(input_size,output_size)
        requires_grad:是否在反向传播中计算权重
        bias：是否设置偏置
        '''

        self.requires_grad = requires_grad
        self.bias = bias

        # 正态分布随机初始化参数矩阵W，乘修正值 2/（sqrt(n)
        # He初始化，使用ReLu
        W = np.random.randn(*shape) * (2 / shape[0] ** 0.5)
        # W = W = np.random.randn(*shape)
        
        # 把W封装到parameter 类里面
        self.W = Parameter(W,self.requires_grad)
        # 根据bias 加入偏置
        self.b = Parameter(np.zeros(shape[-1]),self.requires_grad) if self.bias else None


    def forward(self, x):
        if self.requires_grad: self.x = x

        # a_{ik} = \sum_{j}^{C} x_{ij} w{jk}
        # 用爱因斯坦求和函数，据说矩阵计算效率更高
        a = np.einsum('ij,jk -> ik',x,self.W.data)
        # a = np.dot(x,self.W.data)
        if self.bias: a+= self.b.data

        return a
    
    def backward(self,eta):
        if self.requires_grad:
            batch_size = eta.shape[0]

            #dW{ik} = \frac {1}{N} \sum_{j}{C} x{ji} da_{jk}
            self.W.grad = np.einsum('ji,jk-> ik', self.x, eta) / batch_size

            if self.bias: self.b.grad = np.einsum('i... ->...',eta, optimize=True) / batch_size

            # 在 numpy 版本 1.12.0 之后，einsum加入了 optimize 参数，
            # 用来优化 contraction 操作，对于 contraction 运算部分，
            # 操作的数组包含三个或三个以上，
            # optimize 参数设置能提高计算效率，减小内存占比；
            
            # dz_{ik} = \sum_{j}{C} da{ij} w_{kj}

            return np.einsum('ij,kj- >ik', eta, self.W.data, optimize=True)