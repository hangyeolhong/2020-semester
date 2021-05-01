# coding: utf-8
# 2020/인공지능/final/B711216/홍한결
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # t=1인 값만 남는다.


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()  # 원본은 건드리지 않게
        out[self.mask] = 0  # x<=0이면 0으로 바꿔준다.

        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0  # x<=0이면 0으로 바꿔준다.

        return dx


class CustomActivation:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x, self.dW, self.db = \
            None, None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss, self.y, self.t = \
            None, None, None

    # x는 activation된 값
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = \
            cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5, train_flg=False):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = train_flg

    def forward(self, x):
        if self.train_flg:
            # print("nono")
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            print("맞다")
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    # 여기서 params update 한다.
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class CustomOptimizer:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        # self.m = None
        # self.v = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Model:
    """
    네트워크 모델 입니다.

    """

    def __init__(self, lr=0.01, use_dropout=False, dropout_ratio=0.01, train_flg=False):
        """
        클래스 초기화
        """

        self.params = {}
        self.__init_weight()
        self.__init_layer(dropout_ratio, train_flg)
        self.optimizer = CustomOptimizer(lr)

        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.train_flg = train_flg

    def __init_layer(self, dropout_ratio, train_flg=False):
        """
        레이어를 생성하시면 됩니다.
        """
        self.layers = OrderedDict()  # 순서가 있는 딕셔너리. 딕셔너리에 추가한 순서를 기억한다.
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Dropout1'] = Dropout(dropout_ratio, train_flg)

        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])

        # self.layers['Relu2'] = Relu()
        # self.layers['Dropout2'] = Dropout(dropout_ratio)
        #
        # self.layers['Affine3'] = \
        #     Affine(self.params['W3'], self.params['b3'])
        # self.layers['Relu3'] = Relu()
        # self.layers['Dropout3'] = Dropout(dropout_ratio)
        #
        # self.layers['Affine4'] = \
        #     Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        # 진짜로 weight에 루트 2/n을 곱하니까 정확도가 올라가네 왕.. (He 초기값)
        # input size = output size = 6, hidden size = 5 내맘대로
        self.params['W1'] = np.random.randn(6, 9)
        self.params['b1'] = np.zeros(9)
        self.params['W2'] = np.random.randn(9, 6)
        self.params['b2'] = np.zeros(6)

        # self.params['W1'] = np.random.randn(6, 9) / np.sqrt(6) * np.sqrt(2)
        # self.params['b1'] = np.zeros(9)
        # self.params['W2'] = np.random.randn(9, 6) / np.sqrt(9) * np.sqrt(2)
        # self.params['b2'] = np.zeros(6)

        # self.params['W2'] = np.random.randn(9, 9) / np.sqrt(9) * np.sqrt(2)
        # self.params['b2'] = np.zeros(9)
        # self.params['W3'] = np.random.randn(9, 9) / np.sqrt(9) * np.sqrt(2)
        # self.params['b3'] = np.zeros(9)
        # self.params['W4'] = np.random.randn(9, 6) / np.sqrt(9) * np.sqrt(2)
        # self.params['b4'] = np.zeros(6)

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)
        # return self.last_layer.forward(y, t)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        # grads['W3'] = self.layers['Affine3'].dW
        # grads['b3'] = self.layers['Affine3'].db
        # grads['W4'] = self.layers['Affine4'].dW
        # grads['b4'] = self.layers['Affine4'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val

        # params.pkl을 바이트 형식으로 읽는다.
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        # print("########################################")
        # print(params)
        # print("########################################")

        for key, val in params.items():
            self.params[key] = val

# )a = SoftmaxWithLoss()
# print(a.forward(x)
