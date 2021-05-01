# 4월 13일 실습

# ~Chap 5 함 6,7도.. np로 cnn 만들기
# np의 ndarray

import numpy as np
import matplotlib.pyplot as plt

# exp, -log 함수 그래프 만들기

x = np.arange(0.01, 1, 0.01)
y1 = np.exp(x)
y2 = -np.log(x)

plt.plot(x, y1, label='exp')
plt.plot(x, y2, linestyle='--', label='-log')

plt.title('exp & -log')
plt.legend()
plt.show()

X = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])


# AND
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    if w1 * x1 + w2 * x2 > theta:
        return 1
    else:
        return 0


print("***AND START")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
print("***AND END\n")


def AND_bias(x):
    w = np.array([0.6, 0.6])
    b = -1.0
    return (int(np.sum(w * x) + b >= 0))


def NAND(x):
    w = np.array([-0.6, -0.6])
    b = 1.0
    return (int(np.sum(w * x) + b >= 0))


def OR(x):
    w = np.array([0.6, 0.6])
    b = -0.5
    return (int(np.sum(w * x) + b >= 0))


print("***AND BIAS START")
for i in range(X.shape[0]):
    print(AND_bias(X[i]))
print("***AND BIAS END\n")

print("***NAND BIAS START")
for i in range(X.shape[0]):
    print(NAND(X[i]))
print("***NAND BIAS END\n")

print("***OR BIAS START")
for i in range(X.shape[0]):
    print(OR(X[i]))
print("***OR BIAS END\n")
