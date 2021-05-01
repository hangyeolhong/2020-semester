# coding: utf-8
# 2020/인공지능/final/학번/이름
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


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
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


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


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.15):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=False):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class SoftmaxWithLoss:
    def __init__(self):
        self.loss, self.y, self.t = \
            None, None, None

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


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class CustomOptimizer:
    pass


class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class Model:
    """
    네트워크 모델 입니다.

    """
    hidden_layer_num = 5

    # train_flg = False

    def __init__(self, lr=0.01, weight_decay_lambda=0, use_dropout=True, dropout_ration=0.15, train_flg=False):
        """
        클래스 초기화

        """
        self.params = {}
        self.__init_weight()
        self.__init_layer(use_dropout, dropout_ration)  # layer에 드롭아웃 적용한다고 인자로 넘겨준다.
        self.optimizer = Adam(lr)  # Adam
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout_ration = dropout_ration
        self.train_flg = train_flg

    def __init_layer(self, use_dropout, dropout_ration):
        """
        레이어를 생성하시면 됩니다.

        """
        self.layers = OrderedDict()  # 순서가 있는 딕셔너리. 딕셔너리에 추가한 순서를 기억한다.

        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])

            self.layers['Relu' + str(idx)] = Relu()

            if use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        input_size = 6
        output_size = 6

        self.params['W1'] = np.sqrt(2) * np.random.randn(input_size, 9) / np.sqrt(input_size)
        self.params['b1'] = np.zeros(9)

        for idx in range(2, self.hidden_layer_num + 1):
            self.params['W' + str(idx)] = np.sqrt(2) * np.random.randn(9, 9) / np.sqrt(9)
            self.params['b' + str(idx)] = np.zeros(9)
        # self.params['W2'] = np.sqrt(2) * np.random.randn(12, 12) / np.sqrt(12)
        # self.params['b2'] = np.zeros(12)
        # self.params['W3'] = np.sqrt(2) * np.random.randn(12, 12) / np.sqrt(12)
        # self.params['b3'] = np.zeros(12)
        # self.params['W4'] = np.sqrt(2) * np.random.randn(12, 12) / np.sqrt(12)
        # self.params['b4'] = np.zeros(12)
        # self.params['W5'] = np.sqrt(2) * np.random.randn(12, output_size) / np.sqrt(12)
        # self.params['b5'] = np.zeros(output_size)

        # 마지막 파라미터 초기화
        self.params['W' + str(self.hidden_layer_num + 1)] = np.sqrt(2) * np.random.randn(9, output_size) / np.sqrt(9)
        self.params['b' + str(self.hidden_layer_num + 1)] = np.zeros(output_size)

    # self.params['W2'] = np.sqrt(2) * np.random.randn(9, output_size) / np.sqrt(9)
    # self.params['b2'] = np.zeros(output_size)

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        # grads_numerical = self.numerical_gradient(x,t)

        self.optimizer.update(self.params, grads)

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """

        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, self.train_flg)
            else:
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

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        org_loss = self.last_layer.forward(y, t) + weight_decay

        return org_loss

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
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers[
                'Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        # grads['W1'] = self.layers['Affine1'].dW
        # grads['b1'] = self.layers['Affine1'].db
        # grads['W2'] = self.layers['Affine2'].dW
        # grads['b2'] = self.layers['Affine2'].db
        # grads['W3'] = self.layers['Affine3'].dW
        # grads['b3'] = self.layers['Affine3'].db
        # grads['W4'] = self.layers['Affine4'].dW
        # grads['b4'] = self.layers['Affine4'].db
        # grads['W5'] = self.layers['Affine5'].dW
        # grads['b5'] = self.layers['Affine5'].db
        return grads

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        pass
