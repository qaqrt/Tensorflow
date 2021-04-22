
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import AlexNet
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    train_dir = "E:/data_set/flower_data/train/"
    validation_dir = "E:/data_set/flower_data/val/"
    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10
    # data generator with data augmentation
    # 图像预处理方法
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               target_size=(im_height, im_width),
                                                               )
    # 获取样本的个数
    total_train = train_data_gen.n
    # 获取类别的索引
    class_indices = train_data_gen.class_indices
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir, batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  )

    total_val = val_data_gen.n
    # 将标签转换成one-hot编码的形式
    sample_training_images, sample_training_labels = next(train_data_gen)
    model1 = AlexNet(im_height=im_height, im_width=im_width, class_num=5)
    model1.summary()
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                   loss=tf.keras.losses.CategoricalCrossentropy(),  # 对模型softmax处理是false，没有处理是true
                   metrics=["accuracy"])  # 监控指标是准确率

    # 设置下保存模型的参数
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./save_weights/myAlex.h5",
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    )]

    history = model1.fit(x=train_data_gen,
                         steps_per_epoch=total_train // batch_size,
                         epochs=epochs,
                         validation_data=val_data_gen,
                         validation_steps=total_val // batch_size,
                         callbacks=callbacks)

    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    plt.figure()
    plt.plot(range(epochs), train_loss, label="train_loss")
    plt.plot(range(epochs), val_loss, label="val_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")


if __name__ == '__main__':
    main()
