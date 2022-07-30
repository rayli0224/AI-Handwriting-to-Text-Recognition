from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import tensorflow as tf
import numpy as np
import gradio as gr

import cv2
import functools
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
class_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
TARGET_HEIGHT = 28
TARGET_WIDTH = 28

from emnist import extract_training_samples
training_images, training_labels = extract_training_samples("balanced")
from emnist import extract_test_samples
testing_images, testing_labels = extract_test_samples("balanced")

training_images = training_images / 255.0
testing_images = testing_images / 255.0
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)
num_filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(num_filters, kernel_size, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size),
    tf.keras.layers.Conv2D(num_filters, kernel_size, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(47, activation="softmax")
])
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=5)

def class_idx_to_class(class_idx):
    class_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    return class_mapping[class_idx]

def classify(image):
    image = tf.math.divide(image, 255)
    prediction = model.predict(image.reshape(-1, 28, 28, 1))[0]
    return {str(class_idx_to_class(i)): float(prediction[i]) for i in range(47)}

def classify_word(input):
    # final word/string to return 
    classification = ""
    # apply thresholding to make differences between characters and background more obvious
    image = cv2.imwrite('file.jpg', input)
    image = cv2.imread('file.jpg', 0)
    throwaway, threshold = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.imshow(threshold, cmap='gray')

    # find contours and get bounding box
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(threshold, contours, -1, (0, 255, 0), thickness=1)
    boundingBoxes = [cv2.boundingRect(contour) for contour in contours]

    # sort bounding boxes from left to right and top to bottom so characters are read in the correct order
    boundingBoxes=sorted(boundingBoxes, key=functools.cmp_to_key(compare))
    # loop over bounding boxes
    for rect in boundingBoxes:
        # get coordinates from the bounding box
        x, y, w, h = rect
        # only process if size of character is large enough
        if w * h > 30:
            # crop to only have the character
            crop = image[y:y+h, x:x+w]

            # ensure each character is the correct size for our model (28 x 28) by adding padding
            rows = crop.shape[0]
            columns = crop.shape[1]
            paddingX = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else rows
            paddingY = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else columns

           # add padding 
            crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, value=0)

            # convert and resize image to target height and width
            crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

            # format image data to make prediction
            crop = img_to_array(crop)
            char = crop.reshape((-1, 28, 28, 1))
            
            # make prediction, add to classification string
            char = tf.math.divide(char, 255.0)
            prediction = model.predict(char)[0]
            #dictionary = {str(class_mapping[i]): float(prediction[i]) for i in range(47)}
            classification += class_mapping[np.argmax(prediction)]
    return classification

def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]

image = gr.inputs.Image(shape=(28, 28), image_mode="L", invert_colors=True, source="canvas", type="numpy")
label = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(fn=classify_word, inputs=image, outputs=label, capture_session=True)
interface.launch(share=True, debug=True)
