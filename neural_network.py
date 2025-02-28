import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import copy
import cv2

# Dense Layer Class
# Initilise weights and biases
class Layer_Dense:
    def __init__(self, inputs, neurons, weight_lambda_l1=0, weight_lambda_l2=0, bias_lambda_l1=0, bias_lambda_l2=0):
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        # Regularisation strength
        self.weight_lambda_l1 = weight_lambda_l1 # L1 lambda hyperparameter for weight
        self.weight_lambda_l2 = weight_lambda_l2 # L2 lambda hyperparameter for weight
        self.bias_lambda_l1 = bias_lambda_l1 # L1 lambda hyperparameter for bias
        self.bias_lambda_l2 = bias_lambda_l2 # L2 lambda hyperparameter for bias

# NOTE: Remember that inputs represents the number of features.

    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Forward Pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    # inputs means number of features.

# Backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues) #dL/dW
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) #dL/db
        self.dinputs = np.dot(dvalues, self.weights.T) #dL/dX

        # L1 on weights
        if self.weight_lambda_l1 > 0:
            dL1 = np.ones_like(self.weights) # Calculates sign(W) by making all the positive weights 1
            dL1[self.weights < 0] = -1 # Calculates sign(W) by making all the negative weights -1
            self.dweights += self.weight_lambda_l1 * dL1 
            # This last line updates the properties of dL/dW(total) as it follows the equation
            # dL/dW(total) is equal to dL/dW + Lambda * sign(W)

        # L2 on weights
        if self.weight_lambda_l2 > 0:
            self.dweights += 2 * self.weight_lambda_l2 * self.weights # dL/dW(total) = dL/dW(data) + 2 * lambda * W

        # L1 on biases
        if self.bias_lambda_l1 > 0:
            dL1 = np.ones_like(self.biases) 
            dL1[self.biases < 0] = -1 
            self.dbiases += self.bias_lambda_l1 * dL1 
        
        # L2 on biases
        if self.bias_lambda_l2 > 0:
            self.dbiases += 2 * self.bias_lambda_l2 * self.biases

# DROPOUT
class Layer_Dropout:
    def __init__(self, rate): # In this case, the rate would represent how many neurons you want to drop.
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # dividing the self.rate scales the active neurons so that the total sum of activations is higher
        # this is to compensate for the dropped neurons.
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradients on values
        self.dinputs = dvalues * self.binary_mask

class Layer_Input: # Represents the input layer
        # Forward pass
        def forward(self, inputs, training):
            self.output = inputs

# ReLU activation function
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0] = 0 

    def predictions(self, outputs):
        return outputs

# Softmax activation function
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        # Backslash allows for continuation of the code
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jcaobian matrix of the output and 
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues) # dL/dz

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid activation function
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output # dL/dz = dL/da * da/dz

    def predictions(self, outputs):
        return (outputs > 0.5) * 1 

# Linear activation
class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Loss:
    def regularisation_loss(self): # Regularisation loss calculation
        # 0 by default
        regularisation_loss = 0
        
        # Calculate regularisation loss by iterating over all trainable layers
        for layer in self.trainable_layers:
        # L1 regularisation - weights
            if layer.weight_lambda_l1 > 0:
                regularisation_loss += layer.weight_lambda_l1 * np.sum(np.abs(layer.weights))
            # L2 regularisation - weights
            if layer.weight_lambda_l2 > 0:
                regularisation_loss += layer.weight_lambda_l2 * np.sum(layer.weights**2)
            
            # L1 regularisation - biases
            if layer.bias_lambda_l1 > 0:
                regularisation_loss += layer.bias_lambda_l1 * np.sum(np.abs(layer.biases))

            # L2 regularisation - biases
            if layer.bias_lambda_l2 > 0:
                regularisation_loss += layer.bias_lambda_l2 * np.sum(layer.biases**2)

        return regularisation_loss
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    
    def calculate(self, output, y, *, include_regularisation=False):
        sample_losses = self.forward(output, y) # Loss for each sample in the batch
        data_loss = np.mean(sample_losses) # Average loss over all the samples in the batch

        self.accumulated_sum += np.sum(sample_losses) # Total of the sum of all individual sample losses across multiple batches
        self.accumulated_count += len(sample_losses) # Total of the number of samples processed across multiple batches

        if not include_regularisation:
            return data_loss
        
        return data_loss, self.regularisation_loss()
    
    def calculate_accumulated(self, *, include_regularisation=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularisation:
            return data_loss
        
        return data_loss, self.regularisation_loss()
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
# Cross-entropy loss
class Loss_CategoricalCrossEntropy(Loss):
    # Forword pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clipped both sides 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        # Sparse Encoding
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true
            ]

        # One-Hot Encoding
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1 
            )

        # Losses (individual loss)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0]) # Gives the first row and then the number of classes from that first row.
        
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues #dL/dY(pred): loss with respect to the predicted outputs
        # Normalise gradient
        self.dinputs = self.dinputs / samples 

# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        #Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1) # Converts one-hot encoded to sparse

        # Copy so we can safely modify
        self.dinputs = dvalues.copy() # dvalue is dL/dY
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1 # Gives range and correct sparse vector class and then subtract 1 from all
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# BINARY CROSS ENTROPY LOSS
class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1-clipped_dvalues)) / outputs
        # Normalise gradient
        self.dinputs = self.dinputs / samples

# MEAN SQUARED ERROR LOSS    
class Loss_MeanSquaredError(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

# MEAN ABSOLUTE ERROR LOSS 
class Loss_MeanAbsoluteError(Loss):
    # Forward pass
    def forward (self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalise gradient
        self.dinputs = self.dinputs / samples

# SGD WITH MOMENTUM & VANILLA SGD     
class Optimiser_SGD:
    def __init__ (self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Update the momentum and updating the gradient
            layer.weight_momentums = (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights)
            weight_update = layer.weight_momentums
            layer.bias_momentums = (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases)
            bias_update = layer.bias_momentums

        # Vanilla SGD
        else:
            weight_update = -self.current_learning_rate * layer.dweights
            bias_update = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_update
        layer.biases += bias_update

    def post_update_params(self):
        self.iterations += 1     

# ADAM OPTIMISER
class Optimiser_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.beta1 = beta1 # momentum decay
        self.beta2 = beta2 # cache decay
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
    # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)  
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)  

        # Weights
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        
        weight_corrected_momentums = layer.weight_momentums / (1 - self.beta1**(self.iterations + 1))
        weight_corrected_cache = layer.weight_cache / (1 - self.beta2**(self.iterations + 1))

        layer.weights -= self.current_learning_rate / (np.sqrt(weight_corrected_cache) + self.epsilon) * weight_corrected_momentums

        # Biases
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2
        
        bias_corrected_momentums = layer.bias_momentums / (1 - self.beta1**(self.iterations + 1))
        bias_corrected_cache = layer.bias_cache / (1 - self.beta2**(self.iterations + 1))

        layer.biases -= self.current_learning_rate / (np.sqrt(bias_corrected_cache) + self.epsilon) * bias_corrected_momentums 

    def post_update_params(self):
        self.iterations += 1             

# RMSPROP OPTIMISER
class Optimiser_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)      

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.weights += -(self.current_learning_rate / (np.sqrt(layer.weight_cache) + self.epsilon)) * layer.dweights

        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        layer.biases += -(self.current_learning_rate / (np.sqrt(layer.bias_cache) + self.epsilon)) * layer.dbiases

    def post_update_params(self):
        self.iterations += 1 

# ADAGRAD OPTIMISER
class Optimiser_Adagrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epilson=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epilson = epilson

    # Call once before any parameter updates
    # Learning rate decay formula 
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients 
        # G = G + (dL/dW)^2
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # parameter = parameter - (learning rate/sqrt(G + small value)) * (dL/dW)
        # self.current_learning_rate is used instead of self.learning_rate for the learning rate decay effect.
        layer.weights += -self.current_learning_rate / (np.sqrt(layer.weight_cache) + self.epilson) * layer.dweights
        layer.biases += -self.current_learning_rate / (np.sqrt(layer.bias_cache) + self.epilson) * layer.dbiases

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
