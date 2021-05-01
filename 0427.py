import sys, os
import pickle
import numpy as np

from dataset.mnist import load_mnist
from functions import sigmoid, softmax

sys.path.append(os.pardir)


class MNIST_Predict():
    def __init__(self):
        self.x_test, self.t_test = self.load_data()
        network = self.init_network()
        self.W1 = network['W1']
        self.W2 = network['W2']
        self.W3 = network['W3']
        self.b1 = network['b1']
        self.b2 = network['b2']
        self.b3 = network['b3']

    def load_data(self):
        (x_train, t_train), (x_test, t_test) = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test

    def init_network(self):
        with open("sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)
        return network

    def predict(self, x):

        a1 = np.dot(x, self.W1) + self.b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.W3) + self.b3
        y = softmax(a3)

        return y


mnist = MNIST_Predict()
mnist.load_data()
mnist.init_network()


num = 5 # 10000개 중 5개만
count = 0
for i in range(num):
    y = mnist.predict(mnist.x_test[i])
    print(y)    # 확률값이 프린트 됨
    print(np.argmax(y)) # 그 중 가장 큰 확률의 인덱스
