import sys, os
import numpy as np
from dataset.mnist import load_mnist
from knn import KNN

sys.path.append(os.pardir)

# 이미지를 1차원 배열로 읽고, 원소가 0~1 의 값을 갖도록 한다.
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True)

# 레이블 이름
label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# test data (10000개) 에서 100개 랜덤으로 뽑기
test_size = 100
sample = np.random.randint(0, x_test.shape[0], test_size)

# accuracy 측정을 위해서 초기화한다. computed result 와 실제 값이 일치하는 것의 개수를 저장하는 변수이다.
mv_correct = 0  # majority vote
wmv_correct = 0  # weighted majority vote

# K
K = 3
print("\nWhen K is ...  ", K)
print(" 784개 input 을 그대로 사용한 경우 ")




# 60000개의 train data와 0~9까지의 레이블 이름을 가지고 KNN 클래스의 인스턴스를 만든다.
mnist_result = KNN(x_train, t_train, label_name)




# Majority Vote
print("\n************************************* MAJORITY VOTE START *************************************")

for i in sample:
    computed_class = mnist_result.majority_vote(x_test[i], K)
    print(i, " th data result : ", computed_class, "\tlabel : ", label_name[t_test[i]])
    if computed_class == label_name[t_test[i]]:
        mv_correct += 1

print("\nMajority Vote Accuracy : ", mv_correct / test_size)
print("************************************* MAJORITY VOTE END *************************************\n")




# Weighted Majority Vote
print("********************************** WEIGHTED MAJORITY VOTE START **********************************")

for i in sample:
    computed_class = mnist_result.weighted_majority_vote(x_test[i], K)
    print(i, " th data result : ", computed_class, "\tlabel : ", label_name[t_test[i]])
    if computed_class == label_name[t_test[i]]:
        wmv_correct += 1

print("\nWeighted Majority Vote Accuracy : ", wmv_correct / test_size)
print("********************************** WEIGHTED MAJORITY VOTE END **********************************\n")
