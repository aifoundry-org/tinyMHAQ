import numpy as np

from tinygrad import Tensor

class Accuracy:
    def __init__(self, k=1):
        self.k = k
        
    """
    Numpy implementation of milticlass accuracy metric
    TODO Create a 'jittable' counterpart for the performance.
    """
    def multiclass_accuracy(self, predicitons: np.ndarray, labels: np.ndarray):
        top_k_preds = np.argsort(predicitons, axis=1)[:, -self.k:]
        correct = np.any(top_k_preds == labels[:, None], axis=1) 
        
        return np.mean(correct)