import numpy as np


class KNN:

    def __init__(self, x_train, y_train, names):

        # train 용 data와 레이블, 레이블 이름을 넣어준다.
        self.trainX = x_train
        self.trainY = y_train
        self.nameY = names

    # Euclidean distance
    @staticmethod
    def calc_distance(a, b):
        return np.sqrt(np.sum(np.power(a - b, 2)))

    # train data 중에서 test data 와 가장 가까운 k개의 이웃을 구한다.
    def get_knn(self, test, k):

        dist_arr = np.zeros(len(self.trainX))  # len(self.trainX) = 60000

        # test data(1개) 와 train data(60000개) 간의 거리를 구한다. (총 60000번 계산)
        for i in range(len(self.trainX)):
            dist_arr[i] = self.calc_distance(test, self.trainX[i])

        # KNN
        sorted_dist = np.sort(dist_arr)[:k]  # sorted_dist 는 거리가 작은 순으로 정렬한 결과를 갖는다.

        # KNN
        sorted_index = np.argsort(dist_arr)[:k]  # sorted_index 는 거리가 작은 순으로 정렬 & 데이터 index를 갖는다.

        # labels 는 index 리스트의 각 값에 해당하는 레이블 값(0,1,2)을 갖는다.
        # 원소는 K개, 0으로 초기화
        labels = np.zeros(k)

        for i in range(k):
            labels[i] = self.trainY[sorted_index[i]]

        # knn 을 구한 뒤 각각의 label 을 담고있는 ndarray 와, 이웃들을 거리가 가까운 순으로 정렬한 distance ndarray 를 반환한다.
        return labels, sorted_dist

    def majority_vote(self, test, k):

        label, distance = self.get_knn(test, k)

        # 결과에서 majority vote

        # 0 ~ 9 각 레이블에 속한 개수를 저장하는 target 리스트 , 원소는 10개
        label_count_list = np.zeros(10)

        # label 리스트를 돌면서 어떤 label 에 속하면 label_count_list 에서 해당 value 값을 1 증가
        for i in range(label.shape[0]):
            label_count_list[int(label[i])] += 1

        sort = np.argsort(label_count_list)[::-1]  # sort 는 값이 큰(많이 소속된) 순으로 정렬 & 데이터 index 를 갖는다.

        # sort[0]은 test data 의 knn 이 소속된 레이블 중 가장 많이 소속된 레이블
        # 이 값에 해당하는 name 을 리턴한다.
        return self.nameY[sort[0]]

    def weighted_majority_vote(self, test, k):

        label, distance = self.get_knn(test, k)

        # inverse 는 get_knn 으로 받아온 distance 의 역수 값을 저장한다. 원소는 k개.
        inverse = np.zeros(k)

        # weight_sum_list 는 각 레이블들의 weight 를 저장한다. 원소는 10개이고, 합을 저장할 것이므로 0으로 초기화한다.
        weight_sum_list = np.zeros(10)

        # distance 리스트를 돌면서 거리의 역수를 구한다.
        for i in range(distance.shape[0]):
            # 완전 동일한 경우 역수값에 inf 대신 1000을 집어넣는다.
            if distance[i] == 0.:
                inverse[i] = 1000
            else:
                inverse[i] = 1 / distance[i]

        # 모든 역수 값의 합을 구한다 - inverse_sum 변수
        inverse_sum = np.sum(inverse)

        # weight는 각 거리 역수값을 합으로 나눈 값을 저장하는 리스트
        weight = inverse / inverse_sum

        # label 을 돌면서 해당 레이블에 속하면 label_sum_list 의 해당 value 값에 weight 를 더한다.
        for i in range(label.shape[0]):
            weight_sum_list[int(label[i])] += weight[i]

        # sort 는 값이 큰(합한 weight 가 큰) 순으로 정렬 & 데이터의 index 를 갖는다.
        sort = np.argsort(weight_sum_list)[::-1]

        # sort[0]은 test data 의 knn 이 소속된 레이블들의 weight 값의 합이 최대값인 레이블
        # 이 값에 해당하는 name 을 리턴한다.
        return self.nameY[sort[0]]
