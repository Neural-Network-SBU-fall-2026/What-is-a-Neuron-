import numpy as np

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)


# ============================================================
# Input Range
# ============================================================

x = np.linspace(-10, 10, 1000)

# ============================================================
# Calculate Activations
# ============================================================

y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

print("Sigmoid: ", y_sigmoid)
print("Tanh: ", y_tanh)
print("relu: ", y_relu)


# ============================================================
# Plot All Activation Functions Together
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(x, y_sigmoid, label="Sigmoid", linewidth=2)
plt.plot(x, y_tanh, label="Tanh", linewidth=2)
plt.plot(x, y_relu, label="ReLU", linewidth=2)

plt.axhline(0, linewidth=0.8, color='black')
plt.axvline(0, linewidth=0.8, color='black')

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Activation Functions Comparison")

plt.legend()
plt.grid(True)
plt.show()