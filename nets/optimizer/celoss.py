from typing import Any

import numpy as np

from ..layers.activation import Softmax


class CrossEntropyLoss(object):
    def __init__(self) -> None:
        self.classifier = Softmax()
    
    def gradient(self):
        return self.grad
    
    def __call__(self, a, y, requires_acc=True) -> Any:
        '''
        a: 批量的样本输出
        y: 批量的样本真值
        requires_acc: 是否输出正确率
        return 该批样本的评价损失
        '''
        # 网络输出
        a = self.classifier.forward(a)
        
        # 提前计算梯度
        self.grad = a - y
        # 计算loss
        # 加个数，防止log（0）
        epsilon = 1e-15
        loss = -1 * np.einsum('ij,ij->', y, np.log(a+epsilon), optimize=True) / y.shape[0]
        # print('a',a,'y',y)
        # print(sum(a[0]),max(a[0]))
        # print(sum(y[0]))
        # print(loss)
        # raise False
        if requires_acc:
            acc = np.argmax(a,axis=-1) == np.argmax(y, axis=-1)
            return acc.mean(), loss
        
        return loss


