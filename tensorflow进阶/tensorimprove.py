import tensorflow as tf
import  numpy as np

a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])

a1 = tf.concat([a, b],axis=0)#拼接合并
a2 = tf.stack([a, b], axis=0)#插入一个新的维度

a1 = tf.random.normal([4, 8,8])
result = tf.split(a1, num_or_size_splits=2, axis=0)#张量分割，参数是分割的数量
result1 = tf.unstack(a1, axis=0)#切割长度为1

c = tf.ones([2, 2])
c1 = tf.norm(c, ord=1)#计算L1范数
c2 = tf.norm(c, ord=2)#计算l2范数
c3 = tf.norm(c, ord=np.inf)#计算无穷范数

d = tf.random.normal([4, 10])#模型生成概率
d1 = tf.reduce_max(d, axis=0)

out = tf.random.normal([10,12])
out1 = tf.nn.softmax(out, axis=1)#转换为概率
pre = tf.argmax(out, axis=1)#计算预测值

e = tf.constant([1, 3, 4, 7])#第一个句子
e2 = tf.pad(e,[[1,1]])#在第一个维度左边填充一个数据，右边填充一个数据

f = tf.tile(d, [2,1])#数据复制 参数是第一维度复制2份第二个维度不复制

g = tf.range(9)#随机数产生
g1 = tf.maximum(g, 2)#下限为2
g2 = tf.clip_by_value(g, 2, 8)#限制上下限

h = tf.gather(a,[3,4],axis=0)#收集第一个维度的第4和5的数据
h1 = tf.gather_nd(a,[[1, 1,]])#得到第一行第一列的数据

I = tf.random.normal([4, 38, 5])
I1 = tf.boolean_mask(I, [True, False, False, True], axis=0)#取第一个第四个的数据

J = tf.ones([3, 3])
j1= tf.zeros([3, 3])

cond = tf.constant([[True, False, False], [False, True, False], [True, True, False]])#构造采样条件
j2 = tf.where(cond, J, j1)#如果为True，则选择j中数据，如果为false则选择j1中数据,当参数j和j1为空时，则返回cond中为True的索引

indices = tf.constant([[4], [3], [1], [7]])#更新数据的位置
updates = tf.constant([4.4, 3.3, 1.1, 7.7])#更新的数据
k = tf.scatter_nd(indices, updates, [8])
print(k)





