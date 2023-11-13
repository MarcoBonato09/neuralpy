import numpy as np
import neuralpy

class Network:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.layers = []
        
    
    def add_layer(self, num_neurons: int, activation_function: neuralpy.Function):
        num_prev_neurons = self.input_size if not self.layers else self.layers[-1].size
        new_layer = neuralpy.Layer(num_prev_neurons, num_neurons, activation_function)
        self.layers.append(new_layer)
        
        
    def forward_propagation(self, X: np.ndarray):
        for layer in self.layers:
            layer.forward_propagation(X)
            X = layer.activations
        
        
    def gradient_descent(self, X: np.ndarray, Y: np.ndarray, epochs: int, learning_rate: float, loss_function: neuralpy.Function, batch_size: int):
        n = X.shape[1] # Amount of training examples
        batch_indexes = np.arange(0, n, batch_size)
        X_batches = np.hsplit(X, batch_indexes)
        Y_batches = np.hsplit(Y, batch_indexes)
        for iteration in range(epochs):
            for i in range(len(batch_indexes)):
                X, Y = X_batches[i+1], Y_batches[i+1]
                self.forward_propagation(X)
                last_layer = self.layers[-1]
                
                dEda = loss_function.derivative(last_layer.activations, Y)
                dadz = last_layer.activation_function.derivative(last_layer.z_values)
                dEdz = dEda*dadz
                            
                for index in range(len(self.layers)):
                    index = -(index+1)
                    layer = self.layers[index]
                    input_activations = self.layers[index-1].activations if index != -len(self.layers) else X
                    prev_layer = self.layers[index-1] if index != -len(self.layers) else None
                    dEdz = layer.backward_propagation(prev_layer, input_activations, dEdz, n, learning_rate)                

    
    def output(self, X: np.ndarray):
        self.forward_propagation(X)
        return self.layers[-1].activations
    
    
    def get_loss(self, X: np.ndarray, Y: np.ndarray, loss_function: neuralpy.Function):
        Y_hat = self.output(X)
        n = X.shape[1]
        return np.sum(loss_function(Y_hat, Y))/n
    
    
