class Parameter(object):
    def __init__(self, data, requires_grad, skip_decay=False) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.skip_decay = skip_decay

        self.grad = None
    
    @property
    def T(self):
        return self.data.T
        