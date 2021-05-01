import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from logistic_regression import Logistic_Regression_Multiple, Logistic_Regression_Single

iris = load_iris()

X = iris.data  # 150개
Y = iris.target  # 150개, 0,1,2 중 하나에 속한다.
y_name = iris.target_names

                                ### Multiple Class 실험

# train용 data와 test용 data를 120:30 = 8:2로 분리

# 150개의 index 를 shuffle 한다.
index = np.arange(X.shape[0])
np.random.shuffle(index)

# shuffle 된 index 순서대로 shuffled_X, shuffled_Y 를 만든다.
# 이 결과 X, Y가 임의로 섞인 shuffled_X, shuffeld_Y 를 얻을 수 있다.
shuffled_X = X[index]
shuffled_Y = Y[index]

# shuffled_Y one hot encoding
num = np.unique(shuffled_Y, axis=0)
num = num.shape[0]
shuffled_Y = np.eye(num)[shuffled_Y]

# train data 는 0~119까지 120개
X_train = shuffled_X[:120]
Y_train = shuffled_Y[:120]

# test data 는 120~149까지 30개
X_test = shuffled_X[120:]
Y_test = shuffled_Y[120:]


# train data로 Logistic_Regression 클래스의 인스턴스를 만든다.
lg = Logistic_Regression_Multiple(X_train, Y_train)

# x_train data의 0번째 feature로 bias term=1을 추가한다.
# X_train_with_bias는 m*(n+1)의 2차원 배열이다.
X_train_with_bias = np.insert(X_train, 0, 1, axis=1)


epoch = 1000
alpha = 0.01

# train data 로 학습시킨다.
cost_change = lg.learn(X_train_with_bias, epoch, alpha)

# # 학습이 끝나면 weight 가 변한다.
# print("train 종료 후 weight \n", lg.weight)

# predict하기 위해 x_test에 bias를 추가한다.
X_test_with_bias = np.insert(X_test, 0, 1, axis=1)


# test data로 predict
predict_result = lg.predict(X_test_with_bias)
print("Accuracy ", lg.accuracy(predict_result, Y_test))

# 그래프
plt.plot(np.arange(epoch), cost_change)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()

                                    ### Single Class 실험


# 150개의 index 를 shuffle 한다.
index = np.arange(X.shape[0])
np.random.shuffle(index)

# shuffle된 index 순서대로 shuffled_X, shuffled_Y를 만든다.
# 이 결과 X, Y가 임의로 섞인 shuffled_X, shuffled_Y를 얻을 수 있다.
shuffled_X = X[index]
shuffled_Y = Y[index]

# shuffled_Y one hot encoding
num = np.unique(shuffled_Y, axis=0)
num = num.shape[0]
shuffled_Y = np.eye(num)[shuffled_Y]

# train data 는 0~119까지 120개
X_train = shuffled_X[:120]
Y_train = shuffled_Y[:120]

# test data 는 120~149까지 30개
X_test = shuffled_X[120:]
Y_test = shuffled_Y[120:]

# X_train_with_bias.shape=(120,5)
X_train_with_bias = np.insert(X_train, 0, 1, axis=1)

# Y_train target별로 분리, shape=(120,1)
train_target0 = np.array(Y_train).T[0].reshape(120, 1)
train_target1 = np.array(Y_train).T[1].reshape(120, 1)
train_target2 = np.array(Y_train).T[2].reshape(120, 1)

# Y_test target별로 분리, shape=(120,1)
test_target0 = np.array(Y_test).T[0].reshape(30, 1)
test_target1 = np.array(Y_test).T[1].reshape(30, 1)
test_target2 = np.array(Y_test).T[2].reshape(30, 1)

# target별로 클래스 인스턴스를 만든다.
iris_target0 = Logistic_Regression_Single(X_train, train_target0)
iris_target1 = Logistic_Regression_Single(X_train, train_target1)
iris_target2 = Logistic_Regression_Single(X_train, train_target2)


# train data로 학습
epoch = 200
alpha = 0.5
cost_change0 = iris_target0.learn(X_train_with_bias, epoch, alpha)
cost_change1 = iris_target1.learn(X_train_with_bias, epoch, alpha)
cost_change2 = iris_target2.learn(X_train_with_bias, epoch, alpha)

# predict하기 위해 x_test에 bias를 추가한다.
X_test_with_bias = np.insert(X_test, 0, 1, axis=1)


# test data로 predict
predict_result0 = iris_target0.predict(X_test_with_bias)
predict_result1 = iris_target1.predict(X_test_with_bias)
predict_result2 = iris_target2.predict(X_test_with_bias)

# Accuracy
print("Accuracy ", iris_target0.accuracy(predict_result0, test_target0))
print("Accuracy ", iris_target0.accuracy(predict_result1, test_target1))
print("Accuracy ", iris_target0.accuracy(predict_result2, test_target2))

# 그래프
plt.title("target class 0")
plt.plot(np.arange(epoch), cost_change0)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()

plt.title("target class 1")
plt.plot(np.arange(epoch), cost_change1)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()

plt.title("target class 2")
plt.plot(np.arange(epoch), cost_change2)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
