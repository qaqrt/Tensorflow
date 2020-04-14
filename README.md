 Tensorflow
pycharm开发工具，tensorflow版本2.1
1.常用函数：
2.tf.cast(张量名，dtype=数据类型)#强制转换类型
3.tf.reduce_min(张量名)#计算张量维度上的最小值
4.axis=0是纵向操作，axis=1是横向操作
5.with tf.GradientTape() as tape:
6.  若干个计算过程
7.  grade = tape.gradient(函数，对谁求导)
8.tf.ont_hot（待转换数据，depth=几分类）
9.tf.nn.softmax()#激活函数，使分类符合概率分布
