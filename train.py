#!/usr/bin/env python3

import numpy as np

from common import load_input
from common import sigmoid


inputs, labels, labels_1hot, (K, m, n) = load_input('train.csv')

thetas = np.random.rand(K, n)
alpha = 1e-3
iters = int(20_000)

X = inputs
for k in range(K):
    print(f'train classifier {k} with {m} examples for {iters} iterations')
    theta = thetas[k].reshape(n, 1)
    Y = labels_1hot[k].reshape(m, 1)
    for i in range(iters):

        z = X.dot(theta)
        a = sigmoid(z)

        diff = a - Y
        grad = X.T.dot(diff)
        theta -= (alpha / m) * grad

        if i == iters - 1:
            pass
            # TODO: calculate cost
            # j = (1 / m) * np.sum((Y * np.log(a)) + ((1 - Y) * np.log(1 - a)))

    thetas[k,:] = theta.reshape(n)

# TODO: split train/test data set
for k in range(K):
    theta = thetas[k]
    predictions = sigmoid(X.dot(theta)).round()
    comparison = predictions == labels_1hot[k]
    correct = np.sum(comparison.astype(int))
    total = len(comparison)
    accuracy = correct / total
    percentage = 100 * accuracy
    print(f'accuracy for digit {k}: {percentage:.2f}%')

weights_file = 'weights.csv'
np.savetxt(weights_file, thetas, delimiter=',')
print(f'saved weights as CSV to {weights_file}')
