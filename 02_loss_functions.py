import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))
