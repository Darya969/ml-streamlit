import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

import keras
from keras import layers
from keras.models import Sequential

dataset_dir = pathlib.Path("dataset")

image_count = len(list(dataset_dir.glob("*/*.jpg")))
print(f"All: {image_count}")

batch_size = 32
img_width = 180
img_height = 180

train_ds = keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# create model
num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # аугментация
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # регуляризация
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# load the model
model.load_weights("my_flowers_model.weights.h5")

# evaluate the model
loss, acc = model.evaluate(train_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# load image
img = keras.utils.load_img("1695878391_gas-kvas-com-p-kartinki-oduvanchik-19.jpg", target_size=(img_height, img_width))
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# print inference result
print("На изображении скорее всего {} ({:.2f}% вероятность)".format(
    class_names[np.argmax(score)],
    100 * np.max(score)
))