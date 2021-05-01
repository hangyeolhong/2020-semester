import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


""" 행 별로(각 iris data별로) softmax를 적용하는 함수이다. x는 데이터 개수 * feature 개수인 2차원 배열이다. 
    softmax는 0~1의 값을 리턴한다. 
    행의 원소들을 각 class에 속할 확률로 해석할 수 있다. 행의 원소들을 다 합치면 1이다. """


def softmax(x):
    exp_a = np.zeros_like(x)

    # 각 행에 대하여 (각 iris 데이터에 대하여)
    for i in range(x.shape[0]):
        exp_a[i] = np.exp(x[i] - np.max(x[i]))  # overflow를 막기 위해 입력 신호의 최댓값을 빼준다.
        sum_exp_a = np.sum(exp_a[i])  # 현재 행의 합을 구한다.
        exp_a[i] = exp_a[i] / sum_exp_a  # 현재 행에 softmax를 적용한다.
    return exp_a


# softmax로 너무 큰 값을 넣으면 nan 출력됨


# t는 one hot encoding된 상태
# y, t는 shape이 동일하다. (m*t)
# N개의 데이터에 대한 손실함수를 구한다.
def cross_entropy_error(y, t):
    # 1차원이면 2차원으로 바꿔준다.
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # batch_size는 데이터 개수를 의미
    epsilon = 1e-7  # epsilon은 아주 작은 값이다.

    # shape가 같은 것끼리 곱하므로 element wise product 이다. 이 결과 t=1인 값만 남는다. cee.shape=(m*t)이다.
    cee = -t * np.log(y + epsilon)  # np.log(0)=-inf이므로, 이걸 막기 위해 epsilon을 더한 뒤 log를 취한다.

    # N개의 데이터에 대한 손실함수를 다 더하고 N으로 나눈 평균 손실함수를 리턴한다.
    return np.sum(cee) / batch_size


def numerical_gradient(f, x):
    """ returns gradient value of f at x
    x is a vector containing input value """

    # 1차원이면 1*size인 2차원으로 바꿔준다.
    if x.ndim == 1:
        x = x.reshape(1, x.size)

    h = 1e-4
    grad = np.zeros_like(x)  # grad는 x와 형상이 같다.

    # 각 행(j)의 원소(i)에 대하여
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            temp = x[j][i]
            # f(x+h) 계산
            x[j][i] = temp + h
            fxh1 = f(x[j][i])

            # f(x-h) 계산
            x[j][i] = temp - h
            fxh2 = f(x[j][i])

            # (f(x+h) - f(x-h)) / 2h 계산 ::: 이 값이 곧 f를 jth 행의 ith 원소로 미분한 값이 된다.
            grad[j][i] = (fxh1 - fxh2) / (2 * h)
            x[j][i] = temp

    return grad
