# 与网络相关函数
# 包括读取训练数据与测试数据，训练网络、测试网络、保存权重、读取权重
import numpy as np


def train_nn(nn, train_file, out_nodes, iteration):
    with open(train_file, 'r') as f:
        train_data = f.readlines()

    for i in range(iteration):
        for line in train_data:
            line = line.split(',')
            train_input = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01
            truth_value = np.zeros(out_nodes) + 0.01
            truth_value[int(line[0])] = 0.99
            nn.train(train_input, truth_value)


def test_nn(nn, test_file):
    with open(test_file) as f:
        test_data = f.readlines()

    scoreboard = []

    for line in test_data:
        line = line.split(',')
        test_input = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01
        truth_value = int(line[0])
        test_predict = np.argmax(nn.predict(test_input))

        if test_predict == truth_value:
            scoreboard.append(1)
        else:
            scoreboard.append(0)

    accuracy = scoreboard.count(1) / len(scoreboard)

    return accuracy, scoreboard.count(1), len(scoreboard)


def save_weights(nn):
    np.savetxt('weights_wih', nn.wih)
    np.savetxt('weights_who', nn.who)


def load_weights(wih_weights, who_weights):
    wih = np.loadtxt(wih_weights)
    who = np.loadtxt(who_weights)
    return wih, who
