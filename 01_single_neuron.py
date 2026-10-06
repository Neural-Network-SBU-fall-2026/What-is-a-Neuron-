import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, w, b):
        self.weights = w
        self.bias = b

        print("Neuron is constructed successfully! {}, bias:{}".format(w, b))

    def liniar_neuron_function(self, x):
        z = np.dot(self.weights, x) + self.bias

        return(z)

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def binary_cross_entropy(self, y, y_hat):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss


#input data
x = np.array([2.0, 3.0, 1.0])
w = np.array([0.4, -0.2, 0.7])
b = 0.1

neuron1 = Neuron(w, b)
#liniar neuron function
z = neuron1.liniar_neuron_function(x)


y_hat = neuron1.activation_function(z)
print("Prediction:", y_hat)

#actual lable
y = 1
print("Actual label:", y)


loss = neuron1.binary_cross_entropy(y, y_hat)
print("Loss:", loss)


predictions = np.linspace(0.001, 0.999, 500)

losses = [
    neuron1.binary_cross_entropy(1, p)
    for p in predictions
]


plt.plot(predictions, losses)

plt.xlabel("Prediction")
plt.ylabel("Binary Cross Entropy")
plt.title("BCE Loss when y = 1")

plt.grid(True)
plt.show()

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
    loss = neuron1.binary_cross_entropy(1, prediction)

    print(
        f"Prediction: {prediction:.2f} | "
        f"Loss: {loss:.4f}"
    )