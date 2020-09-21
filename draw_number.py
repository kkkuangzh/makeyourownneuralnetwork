# 通过绘图还原数据集数字样子
import numpy as np
import matplotlib.pyplot as plt

with open("mnist_train_100.csv") as f:
    input_file = f.readlines()

all_value = input_file[0].split(',')
image_array = np.asfarray(all_value[1:]).reshape((28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

