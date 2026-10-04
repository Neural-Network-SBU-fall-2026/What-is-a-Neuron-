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

def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross Entropy
    """
    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return np.mean(loss)