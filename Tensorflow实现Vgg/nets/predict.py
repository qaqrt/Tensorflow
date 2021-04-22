# -- coding:utf-8 --
from PIL import Image
import numpy as np
import json
import Vgg.model
img = Image.open("rose.jpg")
img = img.resize((224, 224))
img = np.array(img) / 255
img = (np.expand_dims(img, 0))

try:
    json_file = open("../class_indices.json")
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = Vgg.model.vgg(class_num=5)
model.load_weights("./save_weights/myAlex_8.h5")
result = model.predict(img)
result1 = np.squeeze(result)
pre_class = np.argmax(result)
print(class_indict[str(pre_class)], result1[pre_class])


