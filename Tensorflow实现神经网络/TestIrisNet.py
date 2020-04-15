from sklearn import datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

epoch = 100  # 迭代次数
lr = 0.1  #学习率
loss_all = 0  # 计算4个loss的和

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

#随机打乱数据
np.random.seed(12)
np.random.shuffle(x_data)
np.random.seed(12)
np.random.shuffle(y_data)
tf.random.set_seed(12)


x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
#训练每次喂入一个batch
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(30)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(30)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))



for epoch in range(epoch):
	for step, (x_train, y_train) in enumerate(train_db):
		with tf.GradientTape() as tape:  #记录梯度信息
			y = tf.matmul(x_train,  tf.cast(w1, dtype=x_train.dtype)) + tf.cast(b1, dtype=x_train.dtype)  #神经网络乘加运算
			y = tf.nn.softmax(y)  #使输出y符合概率分布
			y = tf.cast(y, dtype=x_train.dtype)

			y_ = tf.one_hot(y_train, depth=3)#转换为独热码格式
			y_ = tf.cast(y_, dtype=x_train.dtype)
			loss = tf.reduce_mean(tf.square(y_ - y))  #均方误差计算损失函数
			loss_all +=loss.numpy()
		grads = tape.gradient(loss, [w1,b1])
		w1.assign_sub(lr * grads[0])#参数自更新
		b1.assign_sub(lr * grads[1])
	print("Epoch{},loss{}".format(epoch,loss_all/4))
	loss_all = 0

	#测试
	total_correct, total_number = 0, 0  #预测对的样本个数，总样本个数
	for x_test, y_test in test_db:
		#使用更新后的参数进行预测
		y = tf.matmul(x_test,  tf.cast(w1, dtype=x_train.dtype) )+  tf.cast(b1, dtype=x_train.dtype)
		y = tf.nn.softmax(y)
		pred = tf.argmax(y, axis=1)  #返回y中最大的索引
		#将pred转换为y_test的数据类型
		pred = tf.cast(pred, dtype=y_test.dtype)
		#若分类正确，则correct=1，将bool型的结果转换为int型
		correct = tf.cast(tf.equal(pred,y_test), dtype=tf.int32)
		correct = tf.reduce_sum(correct)  #将每个correct加起来
		#将所有batc中的correct数加起来
		total_correct +=int(correct)
		total_number += x_test.shape[0]
	acc = total_correct / total_number
	print("正确率:", acc)