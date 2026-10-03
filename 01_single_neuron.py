import numpy as np

#input data
x = np.array([2.0, 3.0, 1.0])
w = np.array([0.4, -0.2, 0.7])
b = 0.1

#liniar neuron function
z = np.dot(w, x) + b
print("Weighted sum (z):", z)

#activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


y_hat = sigmoid(z)
print("Prediction:", y_hat)