
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import copy
import cv2
from neural_network import *
from model import *

# Loads an input image and then loads the saved model to make a prediction.
photo = r'INSERT PATH HERE'
image_data = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28)) # Resize
image_data = 255 - image_data # Invert image colour before scaling
plt.imshow(image_data, cmap='gray')
plt.show()
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5 # Reshape and scale pixel data

data_path = r'INSERT PATH HERE'
X, y, X_test, y_test = create_data_mnist(data_path)

fashion_mnist_labels = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "boot"
}

model = Model.load('model_name')
predictions = model.predict(image_data)
predictions = model.output_layer_activation.predictions(predictions)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)
