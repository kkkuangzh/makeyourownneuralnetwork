# make_your_own_neutral_network

*对《python神经网络编程》实现、改进与笔记，不调用keras手动实现神经网络*

*some ideas of the book make your own neutral network by Tariq Rashid*

主要由三部分组成：network_class.py定义了一个Network类，包括网络的结构、前向传播过程(predict)与反向传播过程(train)；network_func.py用于处理训练集与数据集，保存与读取权重；main.py定义文件路径，调用函数实现神经网络功能。 另外还包括draw_number.py，用于将手写数字文件像素值转化为图片。

*由于网络限制只上传了两个小型训练集(100)与测试集(10)，完整训练集包含60000个图像，测试集包括10000个图像，可访问网站http://yann.lecun.com/exdb/mnist/*

## 实现过程的一些问题
1. 更新权重时公式为:new_w = old_w - lr*(δE/δw)，但是由于误差对权重求偏导时公式内部有一个负号，因此实际更新权重时old_w += lr * (out_error * out_output * (1 - out_output)).dot(np.transpose(hidden_output))即可
2. 网络的输入输出列表要转化为2维数组，不然更新权重时会发生矩阵相乘纬度错误(因为要转置)。
3. activation_function = lambda x: scipy.special.expit(x)，直接用lambda来定义函数
4. 初始化随机权重时用np.random.rand()记得-0.5，不然很难更新权重，不管训不训练预测结果都为10个1。当然采用正态分布，根据节点链接数确定权重更好。
5. 保存、读取权重时要将矩阵与字符串互相转换，用np.savetxt('file',data)与np.loadtxt('file')可实现

## 测试结果
网络结构选用简单的784-100-10结构，用60000个训练集进行5次迭代，最后在10000个测试集上的结果正确率为96.5%，并将网络权重保存可供下次使用。输入层与隐含层权重weight_wih，隐含层与输出层权重weight_who。

## 改进方向
常规改进方向：
1. 改变隐含层节点数量与隐含层层数
2. 对输入数据进行预处理
3. 改变激活函数
4. 改变学习率
5. 改变迭代次数，较大的迭代次数应对应较小的学习率
6. 多次运行，避免初始权重分布不理想问题
7. 将图像适当旋转(如±10°)以扩充数据集

## wild thoughts
作者Tariq Rashid在书中提到的反向预测，从后往前对网络输入一个标签，查看网络能输出什么样的图像。从结果来看，网络仅通过一个标签确实能够输出大概的数字轮廓，应出现数字的像素像素值会加深，学习到了一些更深层次的曲线，直线等特征。同时，测试有断点的数字也能正常预测，证明网络确实具有一定鲁棒性。


