import numpy as np
from functions import softmax, cross_entropy_error


# class MulLayer:
#     """ Multiplication Layer for Computational Graph """
#
#     def __init__(self):
#         self.x, self.y = None, None
#
#     def forward(self, x, y):
#         self.x = x
#         self.y = y
#
#         return x * y
#
#     def backward(self, dout):
#         dx = dout * self.y
#         dy = dout * self.x
#
#         return dx, dy
#
#
# apple = 100
# appleNum = 2
# tax = 1.1
#
# mul_apple = MulLayer()
# mul_apple_tax = MulLayer()
#
# apple_price = mul_apple.forward(apple, appleNum)
# total_price = mul_apple_tax.forward(apple_price, tax)
#
# print(total_price)
#
# dout = 1
# dapples, dtax = mul_apple_tax.backward(dout)
# dapple, dnum = mul_apple.backward(dapples)
#
# print(dapple, dnum)

# class Relu:
#     def __init__(self):
#         self.mask = None
#
#     def forward(self, x):
#         self.mask = (x <= 0)
#         out = x.copy()  # 원본은 건드리지 않게
#         out[self.mask] = 0  # x<=0이면 0으로 바꿔준다.
#
#         return out
#
#     def backward(self, dout):
#         dx = dout.copy()
#         dx[self.mask] = 0  # x<=0이면 0으로 바꿔준다.
#         return dx


# overflow 막는게 필요 logistic regression때처럼
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))

        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x, self.dW, self.db = \
            None, None, None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWLoss:
    def __init__(self, W):  # W 사실 없어도 된다.
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
