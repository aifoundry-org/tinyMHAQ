import numpy as np

from tinygrad import Tensor


class Accuracy:
    def __init__(self, k=1):
        self.k = k

    """
    Numpy implementation of milticlass accuracy metric
    TODO Create a 'jittable' counterpart for the performance.
    """

    def multiclass_accuracy(self, predictions: np.ndarray, labels: np.ndarray):
        top_k_preds = np.argsort(predictions, axis=1)[:, -self.k :]
        correct = np.any(top_k_preds == labels[:, None], axis=1)

        return np.mean(correct)

    def __call__(self, predictions: Tensor, labels: Tensor):
        return self.multiclass_accuracy(predictions.numpy(), labels.numpy())
