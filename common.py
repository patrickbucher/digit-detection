import numpy as np


def normalize(x):
    return (x - x.mean()) / x.std()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
