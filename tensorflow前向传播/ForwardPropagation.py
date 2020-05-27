#使用截断的正太分布初始化张量
#偏置张量初始化为0
#第一层的参数
import tensorflow as tf

import cv2
with tf.GradientTape() as tape:
	img1 = cv2.imread('image/0.png')
	img = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

	w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))#标准差为0.1，w和b
	b1 = tf.Variable(tf.zeros([256]))

	#第二层的参数
	w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
	b2 = tf.Variable(tf.zeros([128]))

	#第三层参数
	w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
	b3= tf.Variable(tf.zeros([10]))

	#在前向传播中首先将[x,28,28](x张28行28列的图片)改为[x,784]才适合网络的输入
	x = tf.reshape(img, [-1,28*28])
	x = tf.cast(x, tf.float32)#转换数据类型
	#第一层输入层和权重w相乘加上偏置
	h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
	h1 = tf.nn.relu(h1)#relu激活函数

	#第二层计算
	h2 = h1@w2 + b2
	h1 = tf.nn.relu(h2)#relu激活函数

	#第三层计算
	out = h2@w3 + b3
	y_onehot = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
	#将真实值转为ont-hot码计算均方差
	loss = tf.square(y_onehot - out)
	#误差标量
	loss = tf.reduce_mean(loss)

	#计算梯度，需要计算的梯度有 w1, w2, w3, b1, b2, b3

	grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
	lr = 0.01#学习率

	#进行一次的更新
	w1.assign_sub(lr * grads[0])
	b1.assign_sub(lr * grads[1])
	w2.assign_sub(lr * grads[2])
	b2.assign_sub(lr * grads[3])
	w3.assign_sub(lr * grads[4])
	b3.assign_sub(lr * grads[5])
	print(w1)