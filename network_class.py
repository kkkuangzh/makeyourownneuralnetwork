# 定义网络类，包括结构，预测，训练三部分
import numpy as np
import scipy.special


class Network():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 定义节点数，权重矩阵，学习率，激活函数
        self.inode = input_nodes
        self.hnode = hidden_nodes
        self.onode = output_nodes

        # 记得要-0.5，否则不管训不训练结果10个输出全为1
        self.wih = np.random.rand(self.hnode, self.inode) - 0.5
        self.who = np.random.rand(self.onode, self.hnode) - 0.5
        # self.wih = np.random.normal(0.0, pow(self.hnode, -0.5), (self.hnode, self.inode))
        # self.who = np.random.normal(0.0, pow(self.onode, -0.5), (self.onode, self.hnode))

        self.lrate = learning_rate

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, network_input, truth_value):
        # 必须要先转换成二维数组，否则只有一维更新权重时不加转置会出现维度错误
        network_input = np.array(network_input, ndmin=2).T
        truth_value = np.array(truth_value, ndmin=2).T

        hidden_input = self.wih.dot(network_input)
        hidden_output = self.activation_function(hidden_input)
        out_input = self.who.dot(hidden_output)
        out_output = self.activation_function(out_input)

        out_error = truth_value - out_output
        hidden_error = self.who.T.dot(out_error)

        self.who += self.lrate * (out_error * out_output * (1 - out_output)).dot(np.transpose(hidden_output))
        self.wih += self.lrate * (hidden_error * hidden_output * (1 - hidden_output)).dot(np.transpose(network_input))

    def predict(self, network_input):
        # 前向传播
        network_input = np.array(network_input, ndmin=2).T
        hidden_input = self.wih.dot(network_input)
        hidden_output = self.activation_function(hidden_input)
        out_input = self.who.dot(hidden_output)
        out_output = self.activation_function(out_input)

        return out_output
