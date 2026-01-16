import numpy as np

def regular_times(T: float, delta: float) -> np.ndarray:
    # include 0, exclude T for clean differencing
    n = int(T // delta)
    return np.arange(n + 1) * delta
