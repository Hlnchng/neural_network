import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import copy
import cv2
from neural_network import *

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        # Adds together the accumulated sum of matching vaalues and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy 
    
    def calculate_accumulated(self):

        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0 

# For continuous outputs - regression tasks
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compare predictions to the ground truth values
    def compare(self,predictions, y):
        return np.absolute(predictions - y) < self.precision

# For discrete classes - classification tasks
class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1) # Converts one-hot encoding into sparse coding.
        return predictions == y

# =======================================================
"""
Load an IDX file and return its data as a NumPy array.
Byte 1 and Byte 2 is always 0x00 and represents nothing important.
Byte 3 represents the data type and byte 4 represents the number of dimensions.
"""
def load_idx_file(filename):
    with open(filename, 'rb') as f:  # Open the file in binary mode
        # Read the magic number (first 4 bytes) and sets the byte order as big-endian.
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        
        # Check the data type (3rd byte of the magic number)
        data_type = (magic_number >> 8) & 0xFF # Shifts the magic number by 8 bits and & 0xFF only keeps the last byte (3rd byte)
        if data_type != 0x08:  # 0x08 means unsigned byte
            raise ValueError("Only unsigned byte data is supported.")
        
        # Check the number of dimensions (4th byte of the magic number)
        num_dims = magic_number & 0xFF # keeps only the last byte which is the dimension byte.
        
        # Read the shape of the data - reads 4 bytes and converts it into an integer for each dimension.
        """
        For the magic number, which is the first four bytes. Dimension 1, 2 and 3 which contains 4 bytes each
        are presented by number of images (D1), height of image (D2) and width of each image (D3). 
        Hence, [number of images, height of each image, width of each image]
        """
        shape = [int.from_bytes(f.read(4), byteorder='big') for _ in range(num_dims)]
        
        # Read the data
        data = np.frombuffer(f.read(), dtype=np.uint8) # Converts the bytes into an array.
        # np.uint8 means treat byte as a number between 0 and 255. 
        data = data.reshape(shape)
        # np.frombugffer keeps it a flat array so the dimensions need to be reshaped to the correct dimension. 
        return data


def create_data_mnist(data_path):
    # Data_path: Path to the directory containing the dataset files
    # NOTE: This was done based on the fashion MNIST dataset IDX files.
    X = load_idx_file(fr'{data_path}\train-images-idx3-ubyte')
    y = load_idx_file(fr'{data_path}\train-labels-idx1-ubyte')
    X_test = load_idx_file(fr'{data_path}\t10k-images-idx3-ubyte')
    y_test = load_idx_file(fr'{data_path}\t10k-labels-idx1-ubyte')

    # SHUFFLE THE DATA
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # PREPROCESSING THE DATA 
    # Reshape and normalise to [-1, 1]
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # One-hot encode the labels (optional)
    def one_hot_encode(labels, num_classes=10):
        return np.eye(num_classes)[labels]

    y = one_hot_encode(y)
    y_test = one_hot_encode(y_test)

    return X, y, X_test, y_test

# Check the shapes of the data
#print("Training images shape:", X.shape) 
#print("Training labels shape:", y.shape)  
#print("Test images shape:", X_test.shape)     
#print("Test labels shape:", y_test.shape)    

# Visualise the first training image
#plt.imshow(X[5].reshape(28, 28), cmap='gray')
#plt.title(f"Label: {np.argmax(y[5])}")
#plt.show()

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimiser=None, accuracy=None):
        # The * forces a keyword argument.
        if loss is not None:
            self.loss = loss

        if optimiser is not None:
            self.optimiser = optimiser

        if accuracy is not None:
            self.accuracy = accuracy 

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters
    
    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameter of the model to a file
    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Load weights and updates the model with the weights
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # remove data from input layer and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # Remove input, output and dinput properties for each layer
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open the file in binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model output
        output = []

        # Batching technique on of the input
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                start = step*batch_size
                end=(step+1)*batch_size
                batch_X=X[start:end]

            batch_output = self.forward(batch_X, training=False)

            output.append(batch_output)

        return np.vstack(output)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    # Finalise the model
    def finalise(self): # Establishes the structure of the neural network

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # First layer with the previous layer as the input.
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # Hidden layers except for the first and last layer.
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # Final layer with the next object being the loss function.
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                # Last layer is the output activation layer
                self.output_layer_activation = self.layers[i] 

            # adds layer to the list of trainable layers if it contains the 'weights' attribute.
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # Updates the loss object with trainable layers
            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

            if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
                self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None, batch_size=None):
        
        # Initialise accuracy object
        self.accuracy.init(y) 

        # Default step value if no batch_size is not set.
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            
            X_val, y_val = validation_data
            if batch_size is not None: # Using the batching technique for the validation data
                train_steps = len(X) // batch_size
                if train_steps * batch_size < len(X):
                    train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

            for epoch in range(1, epochs+1):
                print(f'epoch: {epoch}')

                # Reset accumulated values in loss and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(train_steps):
                    # Batch size not step then train using one step and full dataset
                    if batch_size is None:
                        batch_X = X
                        batch_y = y

                    else:
                        start = step*batch_size
                        end = (step+1)*batch_size
                        batch_X = X[start:end]
                        batch_y = y[start:end]
                    
                        """
                        steps * batch_size give images in full batches and len(x_val) gives total images.
                        This calculates any leftovers and adds an additional step to calculate any leftover images.
                        This uses the batching technique.
                        """       

                    # Forward pass
                    output = self.forward(batch_X, training=True)
                    
                    # Calculate loss
                    data_loss, regularisation_loss = self.loss.calculate(output, batch_y, include_regularisation=True)
                    loss = data_loss + regularisation_loss
    

                    # Get predictions and calculate accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    accuracy = self.accuracy.calculate(predictions, batch_y)

                    # Perform backward pass
                    self.backward(output, batch_y)

                    # Optimisation and update parameters
                    self.optimiser.pre_update_params()
                    for layer in self.trainable_layers:
                        self.optimiser.update_params(layer)
                    self.optimiser.post_update_params()

                    # Print summary
                    if not step % print_every or step == train_steps - 1:
                        print(f'step: {step}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularisation_loss:.3f}), ' +
                        f'lr: {self.optimiser.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularisation_loss = self.loss.calculate_accumulated(include_regularisation=True)
            epoch_loss = epoch_data_loss + epoch_regularisation_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
                        
            print(f'training, ' +
                f'acc: {epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f} (' +
                f'data_loss: {epoch_data_loss:.3f}, ' +
                f'reg_loss: {epoch_regularisation_loss:.3f}), ' +
                f'lr: {self.optimiser.current_learning_rate}')

    def evaluate(self, X_val, y_val, *, batch_size=None):
            validation_steps = 1

            if batch_size is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(validation_steps):
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                
                else:
                    start = step*batch_size
                    end = (step+1)*batch_size
                    batch_X = X_val[start:end]
                    batch_y = y_val[start:end]

                # Perform the forward pass
                output = self.forward(batch_X, training=False)
                
                # Calculate the loss
                self.loss.calculate(output, batch_y)

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                self.accuracy.calculate(predictions, batch_y)

            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()

            # Print a summary
            print(f'validation, ' +
                    f'acc: {validation_accuracy:.3f}, ' +
                    f'loss: {validation_loss:.3f}')


    # Performs forward pass 
    def forward(self, X, training):

        self.input_layer.forward(X, training)

        # Output from the previous layer is put into the current layer to connect all layers.
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output
    
    # Performs backward pass
    def backward(self, output, y):
        # backwards pass for the activation_softmax_loss_categoricalcrossentropy
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        
        # Regular backwards pass
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)