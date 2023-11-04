import numpy as np
import neuralpy


class Layer:
    def __init__(self, num_prev_neurons: int, num_neurons: int, activation_function: neuralpy.Function):
        self.size = num_neurons
        self.weights = np.random.rand(num_neurons, num_prev_neurons)-0.5
        self.biases = np.random.rand(num_neurons, 1)-0.5
        self.activation_function = activation_function
        
        
    def forward_propagation(self, X: np.ndarray):
        self.z_values = np.dot(self.weights, X) + self.biases
        self.activations = self.activation_function(self.z_values)
        
        
    def backward_propagation(self, prev_layer, X: np.ndarray, dEdz: np.ndarray, n: int, learning_rate: float):
        dEdw = 1/n*np.dot(dEdz, X.T)
        dEdb = 1/n*np.sum(dEdz, axis=1, keepdims=True)
        
        if prev_layer is not None:
            dEdz = np.dot(self.weights.T, dEdz)*prev_layer.activation_function.derivative(prev_layer.z_values)
                
        self.weights -= dEdw*learning_rate
        self.biases -= dEdb*learning_rate
        
        return dEdz