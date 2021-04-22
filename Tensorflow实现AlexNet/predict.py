import numpy as np
import json
import matplotlib.pyplot as plt
from model import AlexNet
from PIL import Image

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True  # 不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction=0.6  # 限制GPU内存占用率
sess=tf.compat.v1.Session(config=config)

im_height = 224
im_width = 224
img = Image.open("./rose.jpg")
img = img.resize((im_height, im_width))
plt.imshow(img)

img = np.array(img) / 255
img = (np.expand_dims(img, 0))

try:
    json_file = open("./class_indices.json", "r")
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet(class_num=5)
model.load_weights("./save_weights/myAlex.h5")
result = model.predict(img)  # 得到的会是含有batch这个维度的数据 [[0.06769567 0.0463397  0.46235803 0.0457924  0.37781426]]
predict_class = np.squeeze(result)  # 去掉batch维度
predict_class = np.argmax(predict_class)
print(class_indict[str(predict_class)], result[0][predict_class])
plt.show()
