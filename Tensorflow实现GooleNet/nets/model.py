# -- coding:utf-8 --
# 引入了inception结构
# 使用了1*1的卷积核进行降维以及映射处理
# 添加两个辅助分类器帮助训练
# 丢弃全连接层，使用平均池化层
from tensorflow.keras import layers, models, Model, Sequential


def GooLeNet(im_height=224, im_width=224, class_num=1000, aux_logits=False) :
    input_image=layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x=layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
    x=layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
    x=layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
    x=layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    x=layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    x=Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    x=Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)

    x=layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    x=Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)

    if aux_logits:
        aux1 = InceptionAux(class_num, name="aux_1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    if aux_logits:
        aux2 = InceptionAux(class_num, name="aux_2")(x)

    x=Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    x=layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)
    x=Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    x=Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    x=layers.AvgPool2D(pool_size=7, name="avgpool_1")(x)

    x=layers.Flatten(name="output_flatten")(x)
    x=layers.Dropout(rate=0.4, name="output_dropout")(x)
    x=layers.Dense(class_num, name="output_dense")(x)

    aux3=layers.Softmax(name="aux_3")(x)
    if aux_logits :
        model=models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])
    else:
        model=models.Model(inputs=input_image, outputs=aux3)
    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")

        self.branch2 = Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch3 = Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),  # caution: default strides==pool_size
            layers.Conv2D(pool_proj, kernel_size=1, activation="relu")])                  # output_size= input_size

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])
        return outputs



class InceptionAux(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")

        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(num_classes)
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(inputs)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 2048
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        x = self.softmax(x)

        return x
