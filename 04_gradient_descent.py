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
