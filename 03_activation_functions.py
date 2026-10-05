import numpy as np

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