from tlnn import TwoLayerNeuralNetwork2
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data  # 150개
Y = iris.target  # 150개, 0,1,2 중 하나에 속한다.
y_name = iris.target_names

# 타겟을 one hot encoding으로 바꾼다.
num = np.unique(Y, axis=0)
num = num.shape[0]
Y = np.eye(num)[Y]

""" train data와 test data를 120:30 = 8:2로 분리 """

# 150개의 index 를 shuffle 한다.
index = np.arange(X.shape[0])
np.random.shuffle(index)

# shuffle 된 index 순서대로 shuffled_X, shuffled_Y 를 만든다.
# 이 결과 X, Y가 임의로 섞인 shuffled_X, shuffeld_Y 를 얻을 수 있다.
shuffled_X = X[index]
shuffled_Y = Y[index]

# train data 는 0~119까지 120개
X_train = shuffled_X[:120]
Y_train = shuffled_Y[:120]

# test data 는 120~149까지 30개
X_test = shuffled_X[120:]
Y_test = shuffled_Y[120:]

# batch_size 설정
batch_size = 70

input_size = 4  # 4로 고정된다.
hidden_size = 7  # hidden layer 개수도 바꿔가면서 진행
output_size = 3  # 3으로 고정된다.

# train data와 각 층의 노드 개수로 tlnn2 클래스의 인스턴스를 만든다.
tn2 = TwoLayerNeuralNetwork2(X_train, Y_train, input_size, hidden_size, output_size)

# 학습
lr = 0.05
epoch = 10000
cost_change, acc_change = tn2.learn(lr, epoch, batch_size)

# accuracy 출력
# cost가 작아지게끔 변화된 parameter와, train, test data를 사용해서 accuracy를 구한다.
print("Training Accuracy = ", tn2.accuracy(X_train, Y_train))
print("Test Accuracy = ", tn2.accuracy(X_test, Y_test))

# 그래프
plt.title("Neural Network - IRIS")
plt.plot(np.arange(epoch), cost_change, label='cost')
plt.plot(np.arange(epoch), acc_change, label='training accuracy')
plt.legend()
plt.xlabel('number of iterations')
plt.show()
