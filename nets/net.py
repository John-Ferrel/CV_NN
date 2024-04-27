from .layers import *


class Net(Layer):
    def __init__(self, layer_configures) -> None:
        self.layers = []
        self.parameters = []
        for config in layer_configures:
            self.layers.append(self.createLayer(config))


    def createLayer(self,config):
        return self.getDefaultLayer(config)
    
    def getDefaultLayer(self,config):
        t = config['type']
        if t == 'linear':
            layer = Linear(**config)
            self.parameters.append(layer.W)
            if layer.bias : self.parameters.append((layer.b))
        
        elif t == 'relu':
            layer = Relu()
        
        elif t == 'softmax':
            layer = Softmax()
        
        elif t == 'sigmoid':
            layer = Sigmoid()

        elif t == 'tanh':
            layer = Tanh()
        
        else:
            raise TypeError
        
        return layer
    
    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, eta):
        for layer in self.layers[::-1]:
            eta = layer.backward(eta)
        return eta

