import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)