# Neural Network from Scratch using only NumPy
A neural network from scratch with no libaries except NumPy. 

This project implements a neural network from scratch using only NumPy based on the book nnfs, without relying on high-level deep learning libraries like TensorFlow or PyTorch. It serves as an educational resource for understanding the fundamental principles of neural networks and deep learning. This neural network was then used to predict images from the fashion MNIST dataset.

This project was created to strengthen my fundemental knowledge about how a neural network should work under the hood.

## Features
- Fully connected feedforward neural network
- Customisable architecture (number of layers and neurons per layer)
- Activation functions: ReLU, Sigmoid, and Softmax
- Forward and backward propagation implementation
- Stochastic gradient descent (SGD) with momentum & Vanilla gradient descent, Adam, Adagrad and RMSprop
- Support for Dropout regularisation
- Loss functions

## Files
`generate_model.py`- generates the actual model from model.py and allows you to modify the hyperparameters of the model. The model is then saved.

`load_model.py` - Loads up the saved model after it has been generated. It also allows you to load in the dataset for the model to make a prediction on.

`model.py` - Puts everything together. Loads in the dataset, establishes the structure of the neural network from neural_network.py, trains the model on the dataset.

`neural_network.py` - is the actual 'engine' of the neural network.
