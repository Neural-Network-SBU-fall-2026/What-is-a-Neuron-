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
# Plot Sigmoid
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(x, y_sigmoid)
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid Activation Function")
plt.grid(True)

# ============================================================
# Plot Tanh
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(x, y_tanh)
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel("x")
plt.ylabel("Tanh(x)")
plt.title("Tanh Activation Function")
plt.grid(True)

# ============================================================
# Plot ReLU
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(x, y_relu)
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.title("ReLU Activation Function")
plt.grid(True)

# ============================================================
# Show all figures at once
# ============================================================

plt.show()