import sys, os
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from logistic_regression import Logistic_Regression_Single

sys.path.append(os.pardir)

# 이미지를 1차원 배열로 읽고, 원소가 0~1 의 값을 갖도록 한다.
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True)

# train용 data와 test용 data를 32000:8000 = 8:2로 분리

# test data (60000개) 에서 랜덤으로 뽑기
train_size = 2000
sample = np.random.randint(0, x_train.shape[0], train_size)

x_train_ = x_train[sample]
t_train_ = t_train[sample]


x_train_with_bias = np.insert(x_train_, 0, 1, axis=1)


# print(y)

# test data (10000개) 에서 랜덤으로 뽑기
test_size = 500
sample2 = np.random.randint(0, x_test.shape[0], test_size)

x_test_ = x_test[sample2]
t_test_ = t_test[sample2]

# t_train one hot encoding
num = np.unique(t_train_, axis=0)
num = num.shape[0]
y = np.eye(num)[t_train_]

train_target0 = np.array(y).T[0].reshape(train_size, 1)
train_target1 = np.array(y).T[1].reshape(train_size, 1)
train_target2 = np.array(y).T[2].reshape(train_size, 1)
train_target3 = np.array(y).T[3].reshape(train_size, 1)
train_target4 = np.array(y).T[4].reshape(train_size, 1)
train_target5 = np.array(y).T[5].reshape(train_size, 1)
train_target6 = np.array(y).T[6].reshape(train_size, 1)
train_target7 = np.array(y).T[7].reshape(train_size, 1)
train_target8 = np.array(y).T[8].reshape(train_size, 1)
train_target9 = np.array(y).T[9].reshape(train_size, 1)

# t_test one hot encoding
num = np.unique(t_test_, axis=0)
num = num.shape[0]
y = np.eye(num)[t_test_]

test_target0 = np.array(y).T[0].reshape(test_size, 1)
test_target1 = np.array(y).T[1].reshape(test_size, 1)
test_target2 = np.array(y).T[2].reshape(test_size, 1)
test_target3 = np.array(y).T[3].reshape(test_size, 1)
test_target4 = np.array(y).T[4].reshape(test_size, 1)
test_target5 = np.array(y).T[5].reshape(test_size, 1)
test_target6 = np.array(y).T[6].reshape(test_size, 1)
test_target7 = np.array(y).T[7].reshape(test_size, 1)
test_target8 = np.array(y).T[8].reshape(test_size, 1)
test_target9 = np.array(y).T[9].reshape(test_size, 1)

# predict하기 위해 x_test에 bias를 추가한다.
x_test_with_bias = np.insert(x_test_, 0, 1, axis=1)

# train data로 클래스의 인스턴스 만들기
mnist_target0 = Logistic_Regression_Single(x_train_, train_target0)
mnist_target1 = Logistic_Regression_Single(x_train_, train_target1)
mnist_target2 = Logistic_Regression_Single(x_train_, train_target2)
mnist_target3 = Logistic_Regression_Single(x_train_, train_target3)
mnist_target4 = Logistic_Regression_Single(x_train_, train_target4)
mnist_target5 = Logistic_Regression_Single(x_train_, train_target5)
mnist_target6 = Logistic_Regression_Single(x_train_, train_target6)
mnist_target7 = Logistic_Regression_Single(x_train_, train_target7)
mnist_target8 = Logistic_Regression_Single(x_train_, train_target8)
mnist_target9 = Logistic_Regression_Single(x_train_, train_target9)


# train data로 학습
epoch = 200
alpha = 0.005
cost_change0 = mnist_target0.learn(x_train_with_bias, epoch, alpha)
cost_change1 = mnist_target1.learn(x_train_with_bias, epoch, alpha)
cost_change2 = mnist_target2.learn(x_train_with_bias, epoch, alpha)
cost_change3 = mnist_target3.learn(x_train_with_bias, epoch, alpha)
cost_change4 = mnist_target4.learn(x_train_with_bias, epoch, alpha)
cost_change5 = mnist_target5.learn(x_train_with_bias, epoch, alpha)
cost_change6 = mnist_target6.learn(x_train_with_bias, epoch, alpha)
cost_change7 = mnist_target7.learn(x_train_with_bias, epoch, alpha)
cost_change8 = mnist_target8.learn(x_train_with_bias, epoch, alpha)
cost_change9 = mnist_target9.learn(x_train_with_bias, epoch, alpha)

print("target 0 train 종료 후 weight \n", mnist_target0.weight)
print("target 1 train 종료 후 weight \n", mnist_target1.weight)
print("target 2 train 종료 후 weight \n", mnist_target2.weight)
print("target 3 train 종료 후 weight \n", mnist_target3.weight)
print("target 4 train 종료 후 weight \n", mnist_target4.weight)
print("target 5 train 종료 후 weight \n", mnist_target5.weight)
print("target 6 train 종료 후 weight \n", mnist_target6.weight)
print("target 7 train 종료 후 weight \n", mnist_target7.weight)
print("target 8 train 종료 후 weight \n", mnist_target8.weight)
print("target 9 train 종료 후 weight \n", mnist_target9.weight)

# test data로 predict
predict_result0 = mnist_target0.predict(x_test_with_bias)
predict_result1 = mnist_target1.predict(x_test_with_bias)
predict_result2 = mnist_target2.predict(x_test_with_bias)
predict_result3 = mnist_target3.predict(x_test_with_bias)
predict_result4 = mnist_target4.predict(x_test_with_bias)
predict_result5 = mnist_target5.predict(x_test_with_bias)
predict_result6 = mnist_target6.predict(x_test_with_bias)
predict_result7 = mnist_target7.predict(x_test_with_bias)
predict_result8 = mnist_target8.predict(x_test_with_bias)
predict_result9 = mnist_target9.predict(x_test_with_bias)

print("target0 predict_result ",predict_result0)

# Accuracy
print("Accuracy ", mnist_target0.accuracy(predict_result0,test_target0))
print("Accuracy ", mnist_target0.accuracy(predict_result1,test_target1))
print("Accuracy ", mnist_target0.accuracy(predict_result2,test_target2))
print("Accuracy ", mnist_target0.accuracy(predict_result3,test_target3))
print("Accuracy ", mnist_target0.accuracy(predict_result4,test_target4))
print("Accuracy ", mnist_target0.accuracy(predict_result5,test_target5))
print("Accuracy ", mnist_target0.accuracy(predict_result6,test_target6))
print("Accuracy ", mnist_target0.accuracy(predict_result7,test_target7))
print("Accuracy ", mnist_target0.accuracy(predict_result8,test_target8))
print("Accuracy ", mnist_target0.accuracy(predict_result9,test_target9))

# 그래프
plt.title("target 0")
plt.plot(np.arange(epoch), cost_change0)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()

plt.title("target 1")
plt.plot(np.arange(epoch), cost_change1)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()

plt.title("target 2")
plt.plot(np.arange(epoch), cost_change2)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 3")
plt.plot(np.arange(epoch), cost_change3)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 4")
plt.plot(np.arange(epoch), cost_change4)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 5")
plt.plot(np.arange(epoch), cost_change5)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 6")
plt.plot(np.arange(epoch), cost_change6)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 7")
plt.plot(np.arange(epoch), cost_change7)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 8")
plt.plot(np.arange(epoch), cost_change8)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()
plt.title("target 9")
plt.plot(np.arange(epoch), cost_change9)
plt.xlabel('number of iterations')
plt.ylabel('cost')
plt.show()