# coding: utf-8
# 2020/인공지능/final/B711216/홍한결
import sys, os
import argparse
import time

sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from AReM import *
from model import *


class Trainer:
    """
    ex) 200개의 훈련데이터셋, 배치사이즈=5, 에폭=1000 일 경우 :
    40개의 배치(배치당 5개 데이터)를 에폭 갯수 만큼 업데이트 하는것.=
    (200 / 5) * 1000 = 40,000번 업데이트.

    ----------
    network : 네트워크
    x_train : 트레인 데이터
    t_train : 트레인 데이터에 대한 라벨
    x_test : 발리데이션 데이터
    t_test : 발리데이션 데이터에 대한 라벨
    epochs : 에폭 수
    mini_batch_size : 미니배치 사이즈
    learning_rate : 학습률
    verbose : 출력여부

    ----------
    """

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 learning_rate=0.01, verbose=True, use_dropout=False, dropout_ratio=0.01):

        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = int(epochs)
        self.batch_size = int(mini_batch_size)
        self.lr = learning_rate
        self.verbose = verbose
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = int(max(self.train_size / self.batch_size, 1))
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # 렌덤 트레인 배치 생성
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 네트워크 업데이트
        self.network.update(x_batch, t_batch)
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            train_acc, _ = self.accuracy(self.x_train, self.t_train)
            test_acc, _ = self.accuracy(self.x_test, self.t_test)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(
                "=== epoch:", str(round(self.current_epoch, 3)), ", iteration:", str(round(self.current_iter, 3)),
                ", train acc:" + str(round(train_acc, 3)), ", test acc:" + str(round(test_acc, 3)),
                ", train loss:" + str(round(loss, 3)) + " ===")
        self.current_iter += 1

    def train(self, train_flg=True):
        for i in range(self.max_iter):
            self.train_step()

        test_acc, inference_time = self.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc) + ", inference_time:" + str(inference_time))

    def accuracy(self, x, t):

        if t.ndim != 1: t = np.argmax(t, axis=1)  # 원 핫 인코딩이어도 False로 여기서 바뀌네.
        acc = 0.0
        start_time = time.time()

        for i in range(int(x.shape[0] / self.batch_size)):
            tx = x[i * self.batch_size:(i + 1) * self.batch_size]
            tt = t[i * self.batch_size:(i + 1) * self.batch_size]
            y = self.network.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        inference_time = (time.time() - start_time) / x.shape[0]

        return acc / x.shape[0], inference_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py --help 로 설명을 보시면 됩니다."
                                                 "사용예)python train.py --sf=myparam --epochs=10")
    parser.add_argument("--sf", required=False, default="params.pkl", help="save_file_name")
    parser.add_argument("--epochs", required=False, default=20, help="epochs : default=20")
    parser.add_argument("--mini_batch_size", required=False, default=100, help="mini_batch_size : default=100")
    parser.add_argument("--learning_rate", required=False, default=0.01, help="learning_rate : default=0.01")
    args = parser.parse_args()

    # 데이터셋 탑재
    (x_train, t_train), (x_test, t_test) = load_AReM(one_hot_label=True)
    # one hot label=True로 해야 t가 2차원이 되고 t.shape=y.shape이 된다.
    """ x_train = 25055 * 6
        t_train = 25055
        x_test = 8351 * 6
        t_test = 8351 """

    # 모델 초기화
    network = Model(float(args.learning_rate), False, 0.2, train_flg=True)

    # 트레이너 초기화
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=args.epochs, mini_batch_size=args.mini_batch_size,
                      learning_rate=args.learning_rate, verbose=True, use_dropout=False, dropout_ratio=0.2)

    # print("train, 학습 전  시작 ########################################")
    # print(network.params)
    #
    # print("train, 학습 전  끝 ########################################")
    # a = SoftmaxWithLoss()
    # print(a.forward(x_train,t_train))
    # print(a.y.shape)
    # print(a.t.shape)
    # print(a.backward(1))

    # 트레이너를 사용해 모델 학습
    trainer.train()

    network.save_params(args.sf)

    # print("train, 학습 후  시작 ########################################")
    # print(network.params)
    #
    # print("train, 학습 후  끝 ########################################")
    # network.load_params(args.sf)

    # print("########################################")
    # print(network.params)
    # print("########################################")
    # 파라미터 보관
    print("Network params Saved ")
    print(int(x_train.shape[0] / 100))  # 250

"""
# 그래프를 따로 만들어야겠구나.
    # print(len(trainer.train_acc_list))
    # print(trainer.max_iter)
    plt.title("train cost result")
    plt.plot(np.arange(trainer.max_iter), trainer.train_loss_list, label='cost')
    plt.legend()
    plt.xlabel('number of iterations')
    plt.show()


    plt.title("train accuracy result")
    plt.plot(np.arange(trainer.epochs), trainer.train_acc_list, label='training accuracy')
    plt.legend()
    plt.xlabel('number of iterations')
    plt.show()
"""