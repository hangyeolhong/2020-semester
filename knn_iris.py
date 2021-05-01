import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from knn import KNN

iris = load_iris()

X = iris.data  # 150개
Y = iris.target  # 150개, 0,1,2 중 하나에 속한다.
y_name = iris.target_names

# train 용 data 와 test 용 data 분리
for_test = np.array([(i % 15 == 14) for i in range(Y.shape[0])])  # index % 15 = 14 만 True
for_train = ~for_test  # index % 15가 0~14인 것만 True

# print(for_test)
# print(for_train)

X_train = X[for_train]  # 140개
Y_train = Y[for_train]  # 140개

X_test = X[for_test]  # 150개 중 index % 15 = 14인 것만 있으므로 10개
Y_test = Y[for_test]  # 150개 중 index % 15 = 14인 것만 있으므로 10개

# print(X_train)
# print(X_test)
# print(Y_test)

# KNN 클래스의 인스턴스 생성
knn_result = KNN(X_train, Y_train, y_name)

# K
K = 10
print("\nWhen K is ...  ", K)

# Majority Vote
print("\n************************************* MAJORITY VOTE START *************************************")

# 10개의 test data 에 대하여 majority_vote 한 결과를 실제 데이터와 비교하여 출력한다.
for i in range(Y_test.shape[0]):

    computed_class = knn_result.majority_vote(X_test[i], K)
    true_class = y_name[Y_test[i]]

    if computed_class == true_class:
        result = 'Success'
    else:
        result = 'Fail'

    print("Test Data: ", i, " Computed class: ", computed_class, ",\tTrue class: ",
          true_class, ",\tResult :", result)

print("************************************* MAJORITY VOTE END *************************************\n")

# Weighted Majority Vote
print("********************************** WEIGHTED MAJORITY VOTE START **********************************")

# 10개의 test data 에 대하여 weighted_majority_vote 한 결과를 실제 데이터와 비교하여 출력한다.
for i in range(Y_test.shape[0]):

    computed_class = knn_result.weighted_majority_vote(X_test[i], K)
    true_class = y_name[Y_test[i]]

    if computed_class == true_class:
        result = 'Success'
    else:
        result = 'Fail'

    print("Test Data: ", i, " Computed class: ", computed_class,
          ",\tTrue class: ", true_class, ",\tResult :", result)

print("********************************** WEIGHTED MAJORITY VOTE END **********************************\n")

# x1_min, x1_max = X[:, 0].min() - .5, X[:,0].max() + .5
# x2_min, x2_max = X[:, 1].min() - .5, X[:,1].max() + .5
#
# plt.figure(2,figsize=(8,6))
# plt.scatter(X[:,0], X[:,1],c='y',cmap=plt.cm.Set1, edgecolor='k')
#
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(x1_min,x1_max)
# plt.ylim(x2_min,x2_max)
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
