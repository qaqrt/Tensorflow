# Tensorflow
pycharm开发工具，tensorflow版本2.1
常用函数：
tf.cast(张量名，dtype=数据类型)#强制转换类型
tf.reduce_min(张量名)#计算张量维度上的最小值
axis=0是纵向操作，axis=1是横向操作
with tf.GradientTape() as tape:
  若干个计算过程
	grade = tape.gradient(函数，对谁求导)
