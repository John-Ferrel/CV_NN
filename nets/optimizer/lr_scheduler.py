import math


class LRScheduler(object):
    def __init__(self, lr, **config):
        self.lr = lr
        self.config = config
        self.count = 0

        self.update_type = config.get('type') # 使用 .get() 方法以避免 KeyError 
        self.getArgs()

    
    def getArgs(self):
        if self.update_type == 'steplr':
            self.step_size = self.config['step_size']
            self.gamma = self.config['gamma']

        elif self.update_type == 'multisteplr':
            self.milestones = self.config['milestones']
            self.gamma = self.config['gamma']

        elif self.update_type == 'explr':
            self.gamma = self.config['gamma']
        
        else:
            raise TypeError(f"Invalid update type: {self.update_type}. Expected one of 'steplr', 'multisteplr', or 'explr'.")

    

    def lrUpdate(self,loop):
        # 仅当训练轮次变化时，考虑是否更新学习率
        if self.count != loop:
            self.count = loop
            if self.update_type == 'steplr':
                self.lr = self.stepLR(self.count,self.step_size,self.gamma)
            elif self.update_type == 'multisteplr':
                self.lr = self.multiStepLR(self.count,self.milestones,self.gamma)
            elif self.update_type == 'explr':
                self.lr = self.expLR(self.gamma)

            return self.lr
        else:
            return self.lr


    def stepLR(self,cur_step, step_size=10,gamma=0.9):
        # 每30轮降低一次学习率
        if cur_step and cur_step % step_size==0:
            self.lr *= gamma
        print('LRupdate\ncur_step: %d,  lr:%f'%(self.count,self.lr))
        
        return self.lr


    def multiStepLR(self,cur_step,milestones=[5, 15, 30],gamma=0.7):
        if cur_step in milestones:
            self.lr *= gamma
        print('LRupdate\ncur_step: %d,  lr:%f'%(self.count,self.lr))
        
        return self.lr
    

    def expLR(self,gamma=0.99):
        self.lr *= gamma
        print('LRupdate\ncur_step: %d,  lr:%f'%(self.count,self.lr))

        return self.lr


        
        