import numpy as np
from keras import models
import os


def predict(image):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    classes = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    model = models.load_model("model/flowers_sgd_50.h5")
    image = np.reshape(np.asarray(image), (1, 180, 180, 3))
    flower_class = int(np.argmax(model.predict(image)))
    return classes[flower_class]
