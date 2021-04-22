# 首次使用gpu进行网络加速训练
# 使用了Relu激活函数 而不是传统的sigmoid激活函数
# 使用了LBN局部响应归一化
# 在全连接层的前两层中使用了Dropout随机失活神经元
# 过拟合 根本原因是特征维度过多  模型过于复杂， 参数过多 噪声过多

from tensorflow.keras import layers, models, Model, Sequential

def AlexNet(im_height=224, im_width=224, class_num=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # 如果是一个int的话 扩充高和宽 如果是元组（）的话是高和宽 如果是((),())是上下，和左右的元组
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=5, padding="same",activation="relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(class_num)(x)
    predict = layers.Softmax()(x)
    model = models.Model(inputs=input_image, outputs=predict)
    return model

