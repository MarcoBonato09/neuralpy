import numpy as np

class Function:
    def __init__(self, function, derivative_function):
        self.function = function
        self.derivative_function = derivative_function
        
        
    def __call__(self, X, *args):
        return self.function(X, *args)
    
    
    def derivative(self, X, *args):
        return self.derivative_function(X, *args)
        
        
sigmoid = Function(lambda X: 1/(1+np.e**-X), lambda X: 1/(1+np.e**-X)*(1-1/(1+np.e**-X)))
relu = Function(lambda X: np.maximum(0, X), lambda X: X>0)
softmax = Function(lambda X: np.exp(X)/sum(np.exp(X)), lambda X: np.exp(X)/sum(np.exp(X))*(1-np.exp(X)/sum(np.exp(X))))
binary_cross_entropy = Function(lambda a, Y: Y*np.log(a)+(1-Y)*np.log(1-a), lambda a, Y: -Y/a+(1-Y)/(1-a))
mean_squared_error = Function(lambda a, Y: (a-Y)**2, lambda a, Y: 2*(a-Y))


