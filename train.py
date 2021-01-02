#!/usr/bin/env python3

import os

import imageio
import numpy as np
import pandas as pd


def normalize(x):
    return (x - x.mean()) / x.std()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


labels = pd.read_csv('train.csv')

images = labels['path'].apply(lambda p: imageio.imread(p).flatten()).to_numpy()
inputs = np.zeros((len(images), len(images[0])))
for i, image in enumerate(images):
    inputs[i,:] = image

m, n = inputs.shape

labels = labels['digit']
K = len(set(labels.values))

classes = list(range(K))
labels_1hot = {l: (labels == l).astype(int).to_numpy() for l in classes}

thetas = np.random.rand(K, n)
alpha = 1e-3
iters = int(20_000)

X = normalize(inputs)
for k in range(K):
    print(f'train classifier {k} with {m} examplex for {iters} iterations')
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
print(f'saved weights {theta} as CSV to {weights_file}')
