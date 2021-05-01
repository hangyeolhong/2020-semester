import numpy as np
from functions import sigmoid, softmax, cross_entropy_error, numerical_gradient


class TwoLayerNeuralNetwork2():
    """ a neural network with one hidden layer """

    def __init__(self, X, Y, input_size, hidden_size, output_size):
        self.x = X
        self.t = Y

        """ initialize parameters """
        self.params = {}

        # input --> hidden layer: np.random.randn()
        # np.random.randn()은 표준정규분포를 리턴
        self.params['W1'] = np.random.randn(input_size, hidden_size)  # 4*hidden_size
        self.params['b1'] = np.random.randn(hidden_size)  # hidden_size

        # hidden layer --> output layer: np.random.randn()
        self.params['W2'] = np.random.randn(hidden_size, output_size)  # hidden_size * 3
        self.params['b2'] = np.random.randn(output_size)  # 3

        # b1, b2는 1차원 배열이다. numerical_gradient 함수에서 b1, b2를 2차원으로 바꿔서 계산하고 리턴하는데,
        # line 76, 78에서 뺄셈을 할 때 형상이 안 맞아서 오류가 뜨므로 b1, b2를 2차원으로 바꿔줘야 한다.
        self.params['b1'] = self.params['b1'].reshape(1, hidden_size)
        self.params['b2'] = self.params['b2'].reshape(1, output_size)

        self.input_size = input_size
        self.output_size = output_size

    def predict(self, x):
        """ calculate output given input and current parameters: W1, b1, W2, b2 """
        # 활성화 함수는 입력 신호의 총합이 활성화를 일으키는지 정하는 함수이다.

        # input --> hidden layer: sigmoid
        z2 = np.dot(x, self.params['W1']) + self.params['b1']  # 입력신호와 가중치를 곱하고 편향을 더한다.
        a2 = sigmoid(z2)  # 활성화 함수(sigmoid)를 적용한 후 다음 층의 입력신호로 넘겨준다.

        # hidden layer --> output: softmax
        # Neural Network의 출력층에서 사용하는 활성화 함수는 항등 함수와 softmax가 있는데, 분류할 때는 softmax를 쓴다.
        z3 = np.dot(a2, self.params['W2']) + self.params['b2']  # 입력신호와 가중치를 곱하고 편향을 더한다.

        y = softmax(z3)  # 활성화 함수(softmax)를 적용한다. 이 출력은 0~1의 값이어서 확률로 해석할 수 있다.

        return y

    def loss(self, x, t):
        """ calculate loss(cost) value of current hypothesis function """
        """ Neural Network 학습할 때 최적의 theta를 구하기 위해 cost function이 필요하다.
            cost가 작아지는 방향으로 학습시키고, cost를 가능한 작게 하는 theta를 찾는다."""
        y = self.predict(x)

        # x로 predict한 결과와, one hot encoding된 타겟 아웃풋으로 CEE를 구한다. y와 t는 shape이 동일하다.
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)  # y.shape=(m,t)
        y = np.argmax(y, axis=1)  # 행 별로 가장 확률이 큰 것의 index를 구한다. 결과는 원소가 m개인 1차원 배열이다.
        t = np.argmax(t, axis=1)  # t는 one hot encoding 되어있으므로 argmax 결과는 타겟(0,1,2 중 하나)을 의미한다. 결과는 원소가 m개인 1차원 배열이다.
        count = 0
        tf_result = (y == t)  # tf_result는 y==t이면 true로, 아니면 false 값을 가진다.

        for i in range(tf_result.shape[0]):
            if tf_result[i] == True:  # predict 결과와 실제 타겟이 일치하면 count 1 증가
                count += 1

        return count / float(x.shape[0])  # 전체 데이터 개수로 나눠준다.

    # 각 매개변수(W1,b1,W2,b2)에 대하여 cost를 미분한 값을 리턴한다.
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)  # loss_W는 손실함수값

        grads = {}  # grads는 딕셔너리 형태이다.
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])  # cost를 W1으로 미분한다.
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])  # cost를 b1으로 미분한다.
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])  # cost를 W2로 미분한다.
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])  # cost를 b2로 미분한다.

        return grads

    def learn(self, lr, epoch, batch_size):
        # 반복할 때마다 cost와 accuracy의 추이를 보여주기 위해 epoch만큼 zero로 초기화한다.
        # 이 두 변수는 그래프를 그릴 때 필요하다.
        costs = np.zeros(epoch)
        train_accuracies = np.zeros(epoch)

        for i in range(epoch):
            # 120개의 train data에서 batch_size만큼 랜덤으로 뽑는다.
            batch_mask = np.random.choice(120,
                                          batch_size)  # 0~119 인덱스 중 batch_size개만큼 임의로 뽑힌다. batch_mask는 숫자로 이루어진 1차원배열이다.
            x_batch = self.x[batch_mask]  # batch_mask에 해당하는 인덱스의 데이터만 뽑아내서 x_batch를 만든다.
            t_batch = self.t[batch_mask]  # 같은 방법으로 t_batch를 만든다.

            # parameter를 cost가 작아지는 방향으로 조정한다.
            grads = self.numerical_gradient(x_batch, t_batch)
            self.params['W1'] -= lr * grads['W1']
            self.params['b1'] -= lr * grads['b1']
            self.params['W2'] -= lr * grads['W2']
            self.params['b2'] -= lr * grads['b2']

            costs[i] = self.loss(x_batch, t_batch)  # 현재 손실함수값을 ith cost로 저장한다.
            train_accuracies[i] = self.accuracy(x_batch, t_batch)  # 현재 accuracy를 ith accuracy로 저장한다.
            print("cost, accuracy: ", costs[i], "   ", train_accuracies[i])

        return costs, train_accuracies
