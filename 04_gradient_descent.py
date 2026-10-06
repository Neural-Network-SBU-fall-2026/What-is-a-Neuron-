import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


# ============================================================
# 1. Function
# ============================================================

def loss_function(w):
    return (w - 3) ** 2

# ============================================================
# 2. Gradient
# ============================================================

def gradient(w):
    return 2 * (w - 3)


# ============================================================
# 3. Initial Parameters
# ============================================================

w = -5.0

learning_rate = 0.1

epochs = 30

# ============================================================
# 4. Store History
# ============================================================

w_history = []
loss_history = []


# ============================================================
# 5. Gradient Descent
# ============================================================

for epoch in range(epochs):

    loss = loss_function(w)

    grad = gradient(w)

    w_history.append(w)
    loss_history.append(loss)

    print(
        f"Epoch: {epoch + 1:02d} | "
        f"w = {w:.6f} | "
        f"gradient = {grad:.6f} | "
        f"loss = {loss:.6f}"
    )

    # Weight update
    w = w - learning_rate * grad


# ============================================================
# 6. Final Result
# ============================================================

print("\n" + "=" * 60)

print(f"Final weight: {w:.6f}")
print(f"Final loss: {loss_function(w):.6f}")

print("=" * 60)

# ============================================================
# 7. Plot Loss Function
# ============================================================

w_values = np.linspace(-6, 8, 500)

loss_values = loss_function(w_values)


plt.figure(figsize=(10, 6))

plt.plot(
    w_values,
    loss_values,
    label="Loss Function"
)

plt.scatter(
    w_history,
    loss_history,
    label="Gradient Descent Steps"
)

plt.xlabel("Weight (w)")
plt.ylabel("Loss")

plt.title("Gradient Descent")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 8. Plot Loss During Training
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss During Training")

plt.grid(True)

plt.show()
