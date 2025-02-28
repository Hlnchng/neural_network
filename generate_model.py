# Allows user to create and train the model

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import copy
import cv2
from neural_network import *
from model import *

# Load data
data_path = r'INSERT PATH HERE'
X, y, X_test, y_test = create_data_mnist(data_path)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimiser=Optimiser_Adam(decay=1e-4),
    accuracy=Accuracy_Categorical()
)

# Finalise the model
model.finalise()

# Train the model
model.train(
    X, y,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=128,
    print_every=100
)

# Retrieve model parameters
parameters = model.get_parameters()

# New model for evaluation
# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss and accuracy objects
# Optimisers not set as model does not need to be trained this time.
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    accuracy=Accuracy_Categorical()
)

# Finalise the model
model.finalise()

# Set model with parameters instead of training it
model.set_parameters(parameters)

# Evaluate the model
model.evaluate(X_test, y_test)

# Save the model
model.save('model_name')