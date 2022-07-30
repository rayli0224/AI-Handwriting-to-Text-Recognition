# import library
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
# import libraries
import tensorflow as tf
import numpy as np
import gradio as gr
# import emnist/balanced dataset
from emnist import extract_training_samples
training_images, training_labels = extract_training_samples("balanced")
from emnist import extract_test_samples
testing_images, testing_labels = extract_test_samples("balanced")
# keras model handles values from 0 to 1, given image pixels are from 0 to 255
training_images = training_images / 255.0
testing_images = testing_images / 255.0
# cnns require color images, given images are grayscale
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)
# parameters for cnns
num_filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)
# defining layers of the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(num_filters, kernel_size, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size),
    tf.keras.layers.Conv2D(num_filters, kernel_size, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(47, activation="softmax")
])
# setting optimization + loss functions for the model
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# training model
model.fit(training_images, training_labels, epochs=5)
# convert predicted class to digit or lowercase letter
def class_idx_to_class(class_idx):
    class_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    return class_mapping[class_idx]
# process sketchpad image and predict class
def classify(image):
    image = tf.math.divide(image, 255)
    prediction = model.predict(image.reshape(-1, 28, 28, 1))[0]
    return {str(class_idx_to_class(i)): float(prediction[i]) for i in range(47)}
# gradio interface
image = gr.inputs.Image(shape=(28, 28), image_mode="L", invert_colors=True, source="canvas", type="numpy")
label = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(fn=classify, inputs=image, outputs=label, capture_session=True)
interface.launch(share=True, debug=True)
