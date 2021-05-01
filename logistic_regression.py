# m = 전체 데이터 수, n = feature 수, t = 타겟 수라고 하자.

import numpy as np


class Logistic_Regression_Multiple:

    def __init__(self, x, y):
        # x의 0번째 feature로 bias term=1을 추가한 것을 trainX로 저장한다. shape=(m,n+1)
        self.trainX = np.insert(x, 0, 1, axis=1)

        # y를 one hot encoding으로 바꾼 것을 trainY로 저장한다. shape=(m,t)
        num = np.unique(y, axis=0)
        num = num.shape[0]
        self.trainY = np.eye(num)[y]

        # shape=(n+1,t)
        self.weight = np.random.randn(self.trainX.shape[1], self.trainY.shape[1])

        # 전체 데이터 수를 m으로 저장한다.
        self.m = len(self.trainY)
        self.eMin = -np.log(np.finfo(type(0.1)).max)

    # sigmoid function
    def sigmoid(self, z):
        zSafe = np.array(np.maximum(z, self.eMin))
        return 1.0 / (1 + np.exp(-zSafe))

    def hypothesis(self, x):
        mt = np.dot(x, self.weight)  # mt.shape= m*(n+1) * (n+1)*t = (m,t)
        return self.sigmoid(mt)

    # class별 cost를 리턴한다. 리턴 값의 shape=(t,)
    def cost(self, x):
        # tempp.shape=(m,t)이다. class별 cost를 얻기 위해, tempp에서 같은 class끼리 sum을 구하고 m으로 나눠야한다(아래의 반복문 수행).
        tempp = -self.trainY * np.log(self.hypothesis(x)) - (1 - self.trainY) * np.log(1 - self.hypothesis(x))

        temp = []

        # 각 class에 대해서 cost를 구한다.
        for i in range(self.trainY.shape[1]):  # 0≤ i <t

            # class=i일 때의 cost값을 구하고 append한다.
            # i번째 열의 값들을 모두 더하고 m으로 나눈다.
            temp.append((np.sum(np.array(tempp).T[i])) / self.m)

        # numpy array로 변환해서 리턴
        return np.array(temp)

    # cost function의 미분값
    # 리턴값의 shape=(n+1,t)
    def gradient_descent(self, x):
        diff = self.hypothesis(x) - self.trainY

        temp = []

        # m*t * m*1
        # 차이값에 x의 j번째 feature 값을 곱하고 axis=0에 대해 sum을 구한다. (i=1부터 m까지 시그마)
        # np.sum 결과 shape=(t,)인 배열이 생긴다.
        for j in range(self.weight.shape[0]):  # 0≤ i <n+1
            temp.append(np.sum((diff * np.array(x).T[j].reshape(self.m, 1)) / self.m, axis=0))

        # 리스트를 numpy array로 바꾸고, 값이 줄어드는 방향으로 만들기 위해 -를 붙이고 리턴한다.
        return -np.array(temp)

    # train data로 학습시키기
    def learn(self, x, epoch, learning_rate):
        # 진행될 때 마다 각 class의 cost값을 보여줄 것이므로, shape=(epoch,t)이다.
        costs = np.zeros((epoch, self.trainY.shape[1]))

        grads = np.zeros((self.weight.shape[0], self.weight.shape[1]))

        for e in range(epoch):
            costs[e] = self.cost(x)  # costs의 e번째 행을 cost값으로 채운다. epoch=e일 때 각 class의 cost값을 의미한다.
            grads = self.gradient_descent(x)  # grads.shape=(n+1,t)
            print("epoch : ", e, "\tcost: ", costs[e])

            # j번째 feature와 곱해지는 weight에 learning rate*grads[j]를 뺸다.
            for j in range(self.weight.shape[0]):  # 0≤ j <n+1
                self.weight[j] += learning_rate * grads[j]
                # grads[j]는 cost를 j번째 feature와 곱해지는 weight로 미분한 값이다.

        return costs

    # test data로 predict
    def predict(self, x):
        result = self.hypothesis(x)

        # a.shape=(m,)
        a = np.zeros(result.shape[0])

        # m개의 데이터에 대해서 hypothesis()적용한 값 중 가장 큰 것의 index를 구한다.
        for i in range(result.shape[0]):
            a[i] = np.argmax(result[i])

        return a

    def accuracy(self, x, y):
        cnt = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                cnt += 1
        return cnt / len(x)


class Logistic_Regression_Single:
    def __init__(self, x, y):
        self.trainX = np.insert(x, 0, 1, axis=1)
        self.trainY = y
        self.weight = np.random.randn(self.trainX.shape[1], 1)  # shape=(n+1,1)
        self.m = self.trainY.shape[0]
        self.eMin = -np.log(np.finfo(type(0.1)).max)

    # sigmoid function
    def sigmoid(self, z):
        zSafe = np.array(np.maximum(z, self.eMin))
        return 1.0 / (1 + np.exp(-zSafe))

    def hypothesis(self, x):
        mt = np.dot(x, self.weight)  # mt.shape= m*(n+1) * (n+1) = (m,)
        return self.sigmoid(mt)

    def cost(self, x):
        # tempp.shape=(m,)이다. sum을 구하고 m으로 나눈다.
        tempp = -self.trainY * np.log(self.hypothesis(x)) - (1 - self.trainY) * np.log(1 - self.hypothesis(x))
        return np.array(np.sum(np.array(tempp)) / self.m)

    # cost function의 미분값
    def gradient_descent(self, x):
        diff = self.hypothesis(x) - self.trainY

        temp = []

        # m*t * m*1
        # 차이값에 x의 j번째 feature 값을 곱하고 axis=0에 대해 sum을 구한다. (i=1부터 m까지 시그마)
        for j in range(self.weight.shape[0]):  # 0≤ j <n+1
            temp.append(np.sum((diff * np.array(x).T[j].reshape(self.m, 1)) / self.m, axis=0))

        # 리스트를 numpy array로 바꾸고, 값이 줄어드는 방향으로 만들기 위해 -를 붙이고 리턴한다.
        return -np.array(temp)

    # train data로 학습시키기
    def learn(self, x, epoch, learning_rate):
        # 진행될 때 마다 각 class의 cost값을 보여줄 것이므로, shape=(epoch,1)이다.
        costs = np.zeros((epoch, self.trainY.shape[1]))

        # grads.shape=(n+1,1)
        grads = np.zeros((self.weight.shape[0], self.weight.shape[1]))

        for e in range(epoch):
            costs[e] = self.cost(x)  # costs의 e번째 행을 cost값으로 채운다. epoch=e일 때의 cost값을 의미한다.
            grads = self.gradient_descent(x)
            print("epoch : ", e, "\tcost: ", costs[e])

            # j번째 feature와 곱해지는 weight에 learning rate*grads[j]를 뺸다.
            for j in range(self.weight.shape[0]):  # 0≤ j <n+1
                self.weight[j] += learning_rate * grads[j]  # grads[j]는 cost를 j번째 feature와 곱해지는 weight로 미분한 값이다.

        return costs

    # test data로 predict
    def predict(self, x):
        result = self.hypothesis(x)

        # a.shape=(m,)
        a = np.zeros(result.shape[0])

        # hypothesis()적용한 값이 0.5이상이면 class에 속하고, 미만이면 속하지 않는다고 예측한다.
        for i in range(result.shape[0]):
            if result[i] >= 0.5:
                a[i] = 1
            else:
                a[i] = 0

        return a.reshape(x.shape[0], 1)

    def accuracy(self, x, y):
        # Accuracy 구하기
        cnt = 0
        for i in range(len(x)):
            if x[i] == y[i]:
                cnt += 1
        return cnt / len(x)
