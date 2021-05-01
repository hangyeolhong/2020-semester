# B711216 홍한결
import numpy as np

y1 = np.array([0.1, 0.05, 0, 0.6, 0, 0.1, 0, 0.4, 0.05, 0])
t1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
t1_label = np.array([3])

y2 = np.array([[0.1, 0.05, 0, 0.6, 0, 0.1, 0, 0.4, 0.05, 0], [0.1, 0.05, 0, 0.06, 0, 0.1, 0, 0.4, 0.5, 0]])
t2 = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
t2_label = np.array([3, 8])


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    epsilon = 1e-7
    cee = -t * np.log(y + epsilon)
    return np.sum(cee) / batch_size


# # epsilon 추가함으로써 nan을 막을 수 있다.
print(cross_entropy_error(y1, t1))
print(cross_entropy_error(y2, t2))

def f1(x):
    return 0.01 * (x ** 2) + 0.1 * x


def numerical_difference(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# x=5,10일 때의 기울기
# print(numerical_difference(f1, 5))  # 0.2와 유사
# print(numerical_difference(f1, 10))  # 0.3과 유사

# t=1일때만 곱해진다.
# t_label을 인덱스처럼 사용
def cross_entropy_error_label(y, t_label):
    epsilon = 1e-7
    if y.ndim == 1:
        t_label = t_label.reshape(1, t_label.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    cee = -np.sum(np.log(y[np.arange(batch_size), t_label])) / batch_size

    return cee


print(cross_entropy_error_label(y1, t1_label))
print(cross_entropy_error_label(y2, t2_label))


# x는 벡터
def numerical_gradient(f, x):
    # returns gradient value of f at x
    # x is a vector containing input value
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        temp = x[i]
        x[i] = temp + h
        fxh1 = f(x)

        x[i] = temp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = temp

    return grad
def f2(x):
    x2 = 0
    for i in range(x.size):
        x2 += x[i] ** 2
    return x2

x = np.array([-3.0, 4.0])

print(numerical_gradient(f2,x))




x4 = np.array([1, 2, 3, 4])
print(f2(x4))

def gradient_descent(f,init_x,lr=0.1,epoch=100):
    x=init_x
    for i in range(epoch):
        grad = numerical_gradient(f,x)
        x-= lr*grad
    return x

# 거의 0에 가까운 값이 출력된다.
print(gradient_descent(f2,x,lr=0.0001,epoch=100))
