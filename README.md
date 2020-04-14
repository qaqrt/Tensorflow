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
10.更新参数的值，要先定义变量w为可更新
   w = tf.Variable(4)
   w.assign_sub(1)#括号里为要自减的内容
11.tf.argmax(张量名，axis=操作轴）#返回张量沿指定方向维度最大值的索引  
12.tf.greater(a,b)#判断a>b为真还是假
   tf.where(条件语句，真返回A，假返回B)
13.np.random.RandomState.rand(2,3)#括号为维度，产生一个2行3列的随机数矩阵
14.np.vstack(数组1，数组2)#将两个数组垂直方向叠加，就变成2维数组
15.tf.reduce_mean(tf.square(y-y1))#损失函数均方误差
16.tf.losses.categorical_crossentropy(y_,y)#损失函数交叉熵
17.tf.nn.softmax_cross_entropy_with_logits(y_,y)#激活函数和交叉熵
18.欠拟合解决方法：1.增加输入特征项，增加网络参数，减少正则化参数
19.过拟合解决方法：1，数据清洗，增大训练集，采用正则化，增大正则化参数
