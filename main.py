from network_class import Network
import network_func

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.2
iteration = 5

# train_file = "mnist_train_100.csv"
# test_file = "mnist_test_10.csv"
train_file = "mnist_train.csv"
test_file = "mnist_test.csv"
wih_weights = 'weights_wih'
who_weights = 'weights_who'

nn = Network(input_nodes, hidden_nodes, output_nodes, learning_rate)

try:
    nn.wih, nn.who = network_func.load_weights(wih_weights, who_weights)
except OSError:
    network_func.train_nn(nn, train_file, output_nodes, iteration)
    network_func.save_weights(nn)

result, right_num, total_num = network_func.test_nn(nn, test_file)
print("Accuracy is ", result)
print("We got", right_num, "right ones out of", total_num)

