import keras
import os
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Rescaling, InputLayer
from keras.utils.vis_utils import plot_model
import pathlib
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    epochs = 25
    optimizer = 'sgd'

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(180, 180),
        batch_size=32)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(180, 180),
        batch_size=32)

    classes = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    model = keras.Sequential()
    model.add(InputLayer((180, 180, 3)))
    model.add(Rescaling(1. / 255))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))

    plot_model(model,
               to_file="model_plot.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, shuffle=True)

    model.save("model/flowers_{0}_{1}.h5".format(str(optimizer), str(epochs)))
