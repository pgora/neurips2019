# This file contains various iterative updaters for use in gradient method

import numpy as np


class IterativeUpdater:
    """Class responsible for performing an iterative update."""

    def update(self, old_point, gradient):
        raise NotImplementedError


class VanillaGradientDescent(IterativeUpdater):
    """Class for vanila gradient descent update"""

    def __init__(self, lr=0.1):
        self._lr = lr

    def update(self, old_point, gradient):
        gradient = gradient(old_point)
        new_point = old_point - self._lr * gradient
        return new_point
    
class MomentumGradientDescent(IterativeUpdater):
    """Class for momentum gradient descent update"""
    
    def __init__(self, lr=0.01, momentum=0.9):
        self._lr = lr
        self._momentum = momentum    
        self._velocity = np.array([])
        
    def update(self, old_point, gradient):
        gradient = gradient(old_point)

        if len(self._velocity) == 0:
            self._velocity = self._lr * gradient
        else:
            self._velocity = self._momentum * self._velocity + self._lr * gradient
            
        new_point = old_point - self._velocity
        return new_point
    
class NesterovMomentumGradientDescent(IterativeUpdater):
    """Class for Nesterov momentum gradient descent update"""
    
    def __init__(self, lr=0.01, momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        
        self._velocity = np.array([])
        
    def update(self, old_point, gradient):
    
        if len(self._velocity) == 0:
            self._velocity = self._lr * gradient(old_point) / self._momentum
        else:
            gradient = gradient(old_point - self._momentum * self._velocity)
            self._velocity = self._momentum * self._velocity + self._lr * gradient
            
        new_point = old_point - self._velocity
        return new_point
    
class RMSPropGradientDescent(IterativeUpdater):
    """Class for RMSProp gradient descent update"""
    
    def __init__(self, lr=0.001, gamma=0.9, epsilon=1e-8):
        self._lr = lr
        self._gamma = gamma
        self._epsilon = epsilon
        
        self._mean_squared_gradients = np.array([])
        
    def update(self, old_point, gradient):
        gradient = gradient(old_point)
        squared_gradient = gradient * gradient
       
        if len(self._mean_squared_gradients) == 0:
            self._mean_squared_gradients = squared_gradient # no gamma coefficient here, as we are just initializing mean gradients
        else:
            self._mean_squared_gradients = self._gamma * self._mean_squared_gradients + (1.0 - self._gamma) * squared_gradient
       
        new_point = old_point - (self._lr / np.sqrt(self._mean_squared_gradients + self._epsilon)) * gradient
         
        return new_point
        
class AdamGradientDescent(IterativeUpdater):
    """Class for Adam gradient descent update"""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        
        self._epsilon = epsilon
        
        self._mean_gradients = np.array([])
        self._mean_squared_gradients = np.array([])

        self.step_no = 1
        
    def update(self, old_point, gradient):

        gradient = gradient(old_point)
        squared_gradient = gradient * gradient
        
        if len(self._mean_gradients) == 0:
            self._mean_gradients = gradient # no beta1 coefficient here, as we are just initializing mean gradients
        else:
            self._mean_gradients = self._beta1 * self._mean_gradients + (1.0 - self._beta1) * gradient
        
        if len(self._mean_squared_gradients) == 0:
            self._mean_squared_gradients = squared_gradient # no beta2 coefficient here, as we are just initializing mean gradients
        else:
            self._mean_squared_gradients = self._beta2 * self._mean_squared_gradients + (1.0 - self._beta2) * squared_gradient
       
        t = self.step_no
        self.step_no += 1
        
        corrected_mean_gradients = self._mean_gradients / (1.0 - pow(self._beta1, t))
        corrected_mean_squared_gradients = self._mean_squared_gradients / (1.0 - pow(self._beta2, t))
        
        new_point = old_point - (self._lr / (np.sqrt(corrected_mean_squared_gradients) + self._epsilon)) * corrected_mean_gradients
        return new_point