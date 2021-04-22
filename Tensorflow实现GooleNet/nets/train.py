# -- coding:utf-8 --
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import nets.model

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main() :
    img_path=os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    img_path=img_path + "data_set/flower_data/"
    train_dir=img_path + "train"
    validation_dir=img_path + "val"

    if not os.path.exists("../save_weights") :
        os.mkdir("../save_weights")

    im_height=224
    im_width=224
    batch_size=32
    epochs=30

    def pre_function(img):
        img = img / 255.
        img = (img - 0.5) * 2.0
        return img

    # data generator with data augmentation
    train_image_generator=ImageDataGenerator(preprocessing_function=pre_function,
                                             horizontal_flip=True)
    validation_image_generator=ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen=train_image_generator.flow_from_directory(directory=train_dir,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             target_size=(im_height, im_width),
                                                             class_mode="categorical")
    total_train=train_data_gen.n

    val_data_gen=validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                target_size=(im_height, im_width),
                                                                batch_size=batch_size,
                                                                class_mode="categorical")
    class_indices=train_data_gen.class_indices
    invert_dict=dict((val, key) for key, val in class_indices.items())

    json_dir=json.dumps(invert_dict, indent=4)
    with open("../class_indices", "w") as json_fle :
        json_fle.write(json_dir)

    model =nets.model.GooLeNet(im_height=im_height, im_width=im_width, class_num=5, aux_logits=True)
    model.summary()

    # use keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer =tf.keras.optimizers.Adam(learning_rate=0.0003)

    train_loss=tf.keras.metrics.Mean(name="train_loss")
    train_accuracy=tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    test_loss=tf.keras.metrics.Mean(name="test_loss")
    test_accuracy=tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels) :
        with tf.GradientTape() as tape :
            aux1, aux2, output=model(images, training=True)
            loss1=loss_object(labels, aux1)
            loss2=loss_object(labels, aux2)
            loss3=loss_object(labels, output)
            loss=loss1 * 0.3 + loss2 * 0.3 + loss3
            gradients=tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, output)

    @tf.function
    def test_step(images, labels) :
        _, _, output=model(images, training=False)
        t_loss=loss_object(labels, output)
        test_loss(t_loss)
        test_accuracy(labels, output)

    best_test_loss=float("inf")
    for epoch in range(1, epochs + 1) :
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step in range(total_train // batch_size) :
            images, labels=next(train_data_gen)
            train_step(images, labels)

        for step in range(total_train // batch_size) :
            test_image, test_labels=next(val_data_gen)
            test_step(test_image, test_labels)

        template="Epoch{},Loss：{} Accuracy：{}，Test Loss：{}，Test Accuracy：{}"
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss :
            best_test_loss=test_loss.result()
            model.save_weights("../save_weights/myGooLeNet.h5")


if __name__ == '__main__' :
    main()
