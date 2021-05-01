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


# shape = 784인 data 를 49등분 해서 픽셀값 > 0 인(배경이 아니고 실제 숫자가 쓰여진 곳) 개수를 구한다. 이것을 result 배열(shape[0]=49)의 원소로 집어넣는다.
# result 는 수정한 데이터가 저장되는 리스트
# x 는 x_train 또는 x_test 이다.
# x 에서 인덱스=index 인 데이터를 handcraft 한다.
def handcrafted(index, x):
    result = np.zeros(49)   # 초기화

    # 49등분하므로 49번 수행
    for i in range(49):
        count = 0   # 실제 숫자가 있는 것들의 개수를 저장
        # j는 행
        for j in range(4):
            # k는 열
            for k in range(4):
                if x[index][28 * j + k + i * 4] > 0:
                    count += 1
        result[i] = count
    return result


# test data (10000개) 에서 100개 랜덤으로 뽑기
test_size = 100
sample = np.random.randint(0, x_test.shape[0], test_size)

# accuracy 측정을 위해서 초기화한다. computed result 와 실제 값이 일치하는 것의 개수를 저장하는 변수이다.
mv_correct = 0  # majority vote
wmv_correct = 0  # weighted majority vote

# K
K = 3
print("\nWhen K is ...  ", K)
print(" handcrafted input feature ")

revised_x_train = []  # train data input feature 를 handcrafted input feature 로 바꾼 결과를 저장하는 리스트이다.

# 60000개의 train data 를 handcrafted 로 가공한다. 그 결과를 revised_x_train 에 추가한다.
for i in range(x_train.shape[0]):  # 총 60000번 수행
    revised_x_train.append(handcrafted(i, x_train))



# 60000개의 handcrafted train data 와 0~9까지의 레이블 이름을 가지고 KNN 클래스의 인스턴스를 만든다.
revised_mnist_result = KNN(revised_x_train, t_train, label_name)



# Majority Vote
print("\n************************************* MAJORITY VOTE START *************************************")

for i in sample:
    computed_class = revised_mnist_result.majority_vote(handcrafted(i, x_test), K)
    print(i, " th data result : ", computed_class, "\tlabel : ", label_name[t_test[i]])
    if computed_class == label_name[t_test[i]]:
        mv_correct += 1

print("\nMajority Vote Accuracy : ", mv_correct / test_size)
print("************************************* MAJORITY VOTE END *************************************\n")



# Weighted Majority Vote
print("********************************** WEIGHTED MAJORITY VOTE START **********************************")

for i in sample:
    computed_class = revised_mnist_result.weighted_majority_vote(handcrafted(i, x_test), K)
    print(i, " th data result : ", computed_class, "\tlabel : ", label_name[t_test[i]])
    if computed_class == label_name[t_test[i]]:
        wmv_correct += 1

print("\nWeighted Majority Vote Accuracy : ", wmv_correct / test_size)
print("********************************** WEIGHTED MAJORITY VOTE END **********************************\n")
