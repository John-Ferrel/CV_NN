import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from nets.net import Net
from nets.optimizer import *
from utils.mnist_reader import load_mnist


def save(parameters,save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_as,**dic)


def load(parameters, file):
    loc_param = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = loc_param[str(i)]

def label2onehot(Y):
    # 把标签转化为one_hot 编码
    y_onehot = np.zeros((len(Y), 10), dtype=np.uint8)
    y_onehot[np.arange(len(Y)), Y] = 1 
    return y_onehot    


def train_vali_spilt(X,Y,ratio=0.7):
    '''
    标准化，划分训练集和验证集
    输入：图像，标签 
    输入：训练集图像，训练集标签，验证集图像，验证集标签
    '''
    data_size = X.shape[0]
    mean = np.mean(X, axis=0)  
  
    # 计算标准差  
    std = np.std(X, axis=0)  
    
    # 标准化数据  
    X_normalized = (X - mean) / std
    # print(X_normalized[0])
    # raise False
    X = X_normalized
    train_size = int(data_size * ratio)
    Y = label2onehot(Y)
    data_zip = list(zip(X,Y))
    np.random.shuffle(data_zip)
    X, Y = map(np.array,zip(*data_zip))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


def plot_loss_acc(path_pic, loss_train,loss_vali,acc_vali):
    '''
    画图函数，可视化训练中的训练集，验证集的loss和验证集的accuracy
    '''
    x = np.arange(1,len(loss_train)+1)
    fig = plt.figure(1)
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    ax1.plot(x,loss_train,color='b',label='train')
    ax1.plot(x,loss_vali,color='r',label='validation')
    leg = ax1.legend()

    ax2 = plt.subplot(2,1,2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy(%)')
    plt.title('Validation Accuracy')
    ax2.plot(x,acc_vali)

    # print(path_pic)
    # raise False
    plt.savefig(path_pic)
    plt.show()







def train(net,loss_fn,path_data,batch_size,optimizer,load_file,save_file,path_pic,epochs=1,new_train=False,requires_pic=True):
    X,Y = load_mnist(path_data,'train')

    # 训练集拆分
    X_train,Y_train, X_vali,Y_vali = train_vali_spilt(X,Y)
    train_size = X_train.shape[0]
    vali_size = X_vali.shape[0]

    # 训练loss 和accuracy
    list_loss_train = []
    list_loss_vali = []
    list_accu_vali = []

    # 验证集指标
    list_loss_vali_down = []



    if not new_train and os.path.isfile(load_file):
        load(net.parameters,load_file)

    for loop in range(epochs):
        i = 0
        # 分批训练
        while i < train_size - batch_size:
            x = X_train[i:i + batch_size]
            y = Y_train[i:i + batch_size]
            # y_onehot = np.zeros((len(y), 10), dtype=np.uint8)
            # y_onehot[np.arange(len(y)), y] = 1 
            i += batch_size

            output = net.forward(x)
            batch_acc, batch_loss = loss_fn(output,y)
            eta = loss_fn.gradient()
            net.backward(eta)
            optimizer.update(loop)

            if i % 50 == 0:
                print("train:\t","loop: %d, batch: %5d, batch acc: %2.1f, batch loss: %.2f" % \
                    (loop, i, batch_acc*100, batch_loss))
        
        list_loss_train.append(batch_loss)

        # 验证
        x_v = X_vali
        y_v = Y_vali
        # y_v_onehot = np.zeros((len(y_v), 10), dtype=np.float32)
        # y_v_onehot[np.arange(len(y_v)), y_v] = 1 

        output_v = net.forward(x_v)
        acc_v, loss_v = loss_fn(output_v,y_v)
        print("validation\t","loop: %d,  vali acc: %2.1f, vali loss: %.2f" % \
            (loop, acc_v*100, loss_v))
        

        list_loss_vali.append(loss_v)
        list_accu_vali.append(acc_v)

        # 根据 验证集分类准确度自动保存参数
        # if acc_v == max(list_accu_vali) and save_file is not None:
        if abs(acc_v - max(list_accu_vali)) < 1e-9 and save_file is not None:
            print('Model update and save')
            save(net.parameters,save_file)
        else:
            print(abs(acc_v - max(list_accu_vali)))
            print('acc_v is not the best',acc_v,max(list_accu_vali))


        # 训练继续判断（验证集损失是否不在改进）
        if loop == 0:
            pass
        else:
            if list_loss_vali[-2] - list_loss_vali[-1] > 1e-5:
                # loss 显著下降
                list_loss_vali_down.append(1)
            else:
                list_loss_vali_down.append(0)

        # 早停法:当验证误差连续几个迭代周期没有下降或反而上升时，就认为模型已经过拟合，应该停止训练
        k = 5
        if loop > 5 and sum(list_loss_vali_down[-5:]) == 0:
            # 连续5轮loss 不显著下降，结束训练
            break
            
        

    
    if requires_pic:
        plot_loss_acc(path_pic,list_loss_train,list_loss_vali,list_accu_vali)
    

    return max(list_accu_vali) * 100


def test(net, loss_fn, path_data, optimizer, load_file):
    X,Y = load_mnist(path_data,'t10k')

    # 标准化X
    mean = np.mean(X, axis=0)   
    std = np.std(X, axis=0)  
    X_normalized = (X - mean) / std
    X = X_normalized

    # onehotY
    Y= label2onehot(Y)

    # load model
    if os.path.isfile(load_file):
        load(net.parameters,load_file)
    else:
        raise TypeError('ERROR!!! Path of ModelFile is illegal!')

    # test 
    output_v = net.forward(X)
    acc_v, _ = loss_fn(output_v,Y)

    print("Test\t acc: %2.1f" % (acc_v*100))

    return acc_v * 100





if __name__ == "__main__":
    '''
    job = 'train' or 'test'
    '''

    job = 'train'
    # job = 'test'



    '''
    模型选择
    隐藏层大小修改：shape：（m,n）;
    激活函数可选：'relu'，'tanh'，'sigmoid'

    '''
    np.random.seed(1234)
    layers = [
        {'type':'linear','shape':(784,200)},
        {'type':'relu'},
        {'type':'linear','shape':(200,200)},
        {'type':'relu'},
        {'type':'linear','shape':(200,100)},
        {'type':'relu'},
        {'type':'linear','shape':(100,10)},
    ]

    '''
    参数查找：
    学习率：lr
    隐藏层大小：在上面模型那里调整
    正则化强度(衰减权值)：decay
    loss函数：交叉熵损失
    学习率策略：
        步长更新： type='steplr', step_size=1,gamma=0.9
        多步长更新：type='multisteplr', milestones=[1,2,3],gamma=0.8
        指数更新：type='explr',gamma=0.98 #下降非常快，gamma需要更接近于1
    优化器：随机梯度下降，当batch_size==1：SGD
            当batch_size==2^k, mBGD
            当batch_size==data_size，BGD
    批数量：batch_size
    迭代训练次数：epochs
    是否输出并保存图片：requires_pic
    '''
    lr = 0.01
    decay = 0.00001

    loss_fn = CrossEntropyLoss()
    net = Net(layers)
    lr_type = 'steplr'
    LRsche = LRScheduler(lr,type=lr_type, step_size=5,gamma=0.9)
    # LRsche = LRScheduler(lr,type='explr',gamma=0.98)
    optimizer = SGD(net.parameters,LRsche,decay)
    batch_size = 256
    epochs = 20
    requires_pic = True


    '''
    header  = {name of parameters,...}
    activations: 'rrr' means relu+relu+relu, t means tanh, s means sigmoid
    '''
    headers = ['job','acti','lr','lr_type','decay','batch_size','epochs','acc']
    hyperParam = {'job':job,'acti':'rrr',\
                'lr':lr,'lr_type':lr_type,'decay':decay,\
                'batch_size':batch_size,'epochs':epochs,'acc':0}
    str_hyperParam = '-'.join([str(v) for v in hyperParam.values()])
    '''
    数据集目录路径: path_data
    参数文件路径：path_param
    '''
    path_data = './data'  
    path_param = './model/MNIST_param.npz'
    path_pic = os.path.join('./report/pics/',str_hyperParam + '.png')
    print(path_pic)
    '''
    是否新的训练，不加载已有参数：new_train
    '''
    new_train = True


    if job == 'train':
        acc = train(net, loss_fn, path_data, batch_size, optimizer, path_param,path_param,path_pic,epochs,new_train,requires_pic)
    elif job == 'test':
        acc = test(net, loss_fn, path_data, optimizer, path_param)
    else:
        TypeError('Choose "train" or "test"')
    

    # 训练结果保存
    path_resSave = './report/result.csv'

    if not os.path.exists(path_resSave):
        with open(path_resSave, mode='w', newline='') as csv_file:  
            writer = csv.DictWriter(csv_file, fieldnames=headers)  
            writer.writeheader() 

    hyperParam['acc'] = acc

    with open(path_resSave, mode='a', newline='') as csv_file:  
        writer = csv.DictWriter(csv_file, fieldnames=headers)  
        writer.writerow(hyperParam)    