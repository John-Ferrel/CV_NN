# CV_NN
PJ1 of CV, a three-layer linear neural network for image classification

## How to use it to train and test

### Environment
python                    3.12.3
numpy                     1.26.4
matplotlib                3.8.4 

### Data 
https://github.com/zalandoresearch/fashion-mnist

### Train
Open the file "train_test.py", set the patameter job to "train", and then run

### Test
Open the file "train_test.py", set the patameter job to "test", and then run

### Parameter
**Net**: the network, including lays and activations\[ReLU,Sigmoid,Tanh\]


**lr, lr_scheduler**: Set the learning rate and the scheduler of updating\[steplr,mulitisteplr,explr\]


**L2_norm_rate(decay)**:In the SGD system optimizer without introducing momentum, weight decay is completely equivalent to L2 regularization.


**batch_size**:Usually 2^k


**epochs**:Set the maximum epochs, and if there are multiple epochs with no significant decrease, the training will be automatically stopped
