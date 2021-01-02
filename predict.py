#!/usr/bin/env python3

import numpy as np

from common import load_input
from common import sigmoid


inputs, labels, labels_1hot, (K, m, n) = load_input('predict.csv')

weights_file = 'weights.csv'
thetas = np.loadtxt(weights_file, delimiter=',')
print(f'loaded weights from {weights_file}')

for i in range(len(inputs)):
    image = inputs[i,:]
    image = image.reshape(image.shape[0], 1)
    label = labels[i]
    print(f'\nimage {i} has label {label}')
    ps = sigmoid(thetas.dot(image))
    top_i = np.argmax(ps)
    top = np.squeeze(ps[top_i])
    print(f'prediction: {top_i} with {top * 100:.3f}%')
    for k in range(len(ps)):
        p = np.squeeze(ps[k])
        print(f'\tP({k}) = {p * 100:7.3f}%')
