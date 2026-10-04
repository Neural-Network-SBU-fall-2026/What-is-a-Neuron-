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




y_true = 1

predictions = np.array([
    0.01,
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
    0.99
])


print("=" * 60)
print("LOSS COMPARISON")
print("=" * 60)

print(f"{'Prediction':<15} {'MSE':<15} {'MAE':<15} {'BCE':<15}")
print("-" * 60)

for prediction in predictions:

    mse = mse_loss(y_true, prediction)
    mae = mae_loss(y_true, prediction)
    bce = binary_cross_entropy(y_true, prediction)

    print(
        f"{prediction:<15.2f}"
        f"{mse:<15.4f}"
        f"{mae:<15.4f}"
        f"{bce:<15.4f}"
    )


print("\n" + "=" * 60)
print("BAD PREDICTION ANALYSIS")
print("=" * 60)

y_true = 1

bad_predictions = [
    0.5,
    0.1,
    0.01,
    0.001
]

for prediction in bad_predictions:

    mse = mse_loss(y_true, prediction)
    mae = mae_loss(y_true, prediction)
    bce = binary_cross_entropy(y_true, prediction)

    print(
        f"Prediction = {prediction:.3f} | "
        f"MSE = {mse:.4f} | "
        f"MAE = {mae:.4f} | "
        f"BCE = {bce:.4f}"
    )