import numpy as np

from .layer import Layer

'''
激活函数的作用：增加非线性因素，解决线性模型表达能力不足的缺陷。
如果激活函数是线性的，神经网络叠再多层也没什么意义，和最基本的线性回归是一模一样的。
'''



class Relu(Layer):
    '''
    优点：
        计算简单高效，相比sigmoid、tanh没有指数运算
        相比sigmoid、tanh更符合生物学神经激活机制
        在正区间不饱和（Does not saturate），解决梯度消失的问题
        收敛速度较快，大约是 sigmoid、tanh 的 6 倍
    缺点：
        输出not zero-centered,0点无梯度
        Dead ReLU Problem：反向传播时，梯度横为零，参数永远不会更新
    '''
    def forward(self, x):
        self.x = x
        return np.maximum(0,x)
    
    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta


# class Sigmoid(Layer):
#     '''
#     缺点：
#         左右两侧都是近似饱和区，导数太小，容易造成梯度消失
#         涉及指数运算
#         not zero-centered：输出值不以零为中心，会导致模型收敛速度慢
#     '''
#     def forward(self, x):
#         self.y = 1 / (1 + np.exp(-x))
#         return self.y

#     def backward(self, eta):
#         # x = 
#         return np.einsum('...,...,...->...', self.y, 1 - self.y, eta, optimize=True)
    
class Sigmoid(Layer):  
      
    def forward(self, x):  
        self.x = x  # 保存输入值，以便在反向传播中使用  
        self.y = 1 / (1 + np.exp(-x))  
        return self.y  
  
    def backward(self, eta):  
        # 计算sigmoid函数的导数  
        sigmoid_derivative = self.y * (1 - self.y)  
        # 计算损失函数关于输入的梯度  
        # 通过逐元素乘法将导数应用于传入的梯度eta  
        dx = sigmoid_derivative * eta  
        return dx
    

class Tanh(Layer):
    '''
    相比sigmoid，收敛速度更快，因为在0附近的线性区域内斜率更大，收敛速度会加快。
    相比sigmoid，tanh的输出均值为0，不存在sigmoid中 
    恒为正或者恒为负的情况。
    相比sigmoid，也存在近似饱和区，而且范围比sigmoid更大。
    '''
    def forward(self,x):
        ex = np.exp(x)
        esx = np.exp(-x)
        self.y = (ex - esx) / (ex + esx)
        return self.y
    
    def backward(self,eta):
         return np.einsum('..,..,...->...', 1-self.y, 1+self.y, eta, optimize=True)
    


class Softmax(Layer):
    def forward(self,x):
        '''
        x.shape = (N,c)
        '''

        # a_{i,j} = \frac{e^{x_{ij}}}{\sum{j}{C} e^{x^{ij}}}
        # 所有元素减去最大值，防止溢出
        v = np.exp(x-x.max(axis=-1,keepdims=True))
        self.a = v/v.sum(axis=-1,keepdims=True)
        return self.a
    
    def backward(self,eta):
        return self.a * (eta - np.einsum('ij,ij->i', eta,self.a,optimize=True))