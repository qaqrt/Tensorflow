import tensorflow as tf
import numpy as np

aa = tf.constant([[[1, 3, 4], [2, 3, 4]], [[4, 5, 6], [1, 3, 4]], [[4, 5, 5], [3, 5, 1]]])#创建一个张量
a = tf.constant(12, dtype=tf.int32)
b = tf.cast(a, dtype=tf.int64)#类型转换
print("a的类型{}    b的类型{}".format(a.dtype, b.dtype))

tf.convert_to_tensor(np.array([[1, 3], [2, 5]]))#从数组中创建张量
b = tf.ones([3,4,1])#创建全为1的3维张量，tf.zeros创建全为0

c = tf.zeros_like(b)#创建于b形状一样的全为0的张量，tf.ones_like


d = tf.fill([2, 2], 99)#创建2行2列，全为99的矩阵

e = tf.random.normal([2, 2])#创建2行2列标准正态分布的张量 注：同理能创建别的分布
e1 = tf.random.normal([2, 2],mean = 1,stddev=2)#创建均值为1，标准差为2的

f = tf.range(1, 10,delta = 2)#创建步长为2的序列

g = tf.random.normal([4, 32, 32, 3])#4张32*32的彩色图片

h = tf.range(96)
h1 = tf.reshape(h, [2, 4, 4, 3])#改变维度
h2 = tf.expand_dims(h1, axis=0)#增加维度
h3 = tf.squeeze(h2, axis=0)#删除维度
h4 = tf.transpose(h3, perm=[0, 2, 1, 3])#交换维度

h = tf.range(4)
h1 = tf.reshape(h,[1,4])
h2 = tf.tile(h1,multiples=[1,2])#列复制一份

h3 = tf.broadcast_to(h1, [2, 32, 4, 4])
#tf.pow(h,2)乘方运算，
#tf.math.log(x)/tf.math.log(10)#换底公式
#tf.matmul(a,b)矩阵相乘的条件是a的倒数第一个列和b的倒数第二个行相等


