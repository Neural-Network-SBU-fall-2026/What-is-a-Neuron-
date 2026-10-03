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

#actual lable
y = 1
print("Actual label:", y)

def binary_cross_entropy(y, y_hat):
    epsilon = 1e-15

    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    return loss


loss = binary_cross_entropy(y, y_hat)
print("Loss:", loss)

print("\nPrediction vs Loss")
print("-" * 30)

predictions = np.array([
    0.01,
    0.10,
    0.30,
    0.50,
    0.70,
    0.90,
    0.99
])

for prediction in predictions:
    loss = binary_cross_entropy(1, prediction)

    print(
        f"Prediction: {prediction:.2f} | "
        f"Loss: {loss:.4f}"
    )