#!/usr/bin/env python3

import os

import imageio
import numpy as np
import pandas as pd


labels = pd.read_csv('labels.csv')
labels['image'] = labels['path'].apply(lambda p: imageio.imread(p))

goals = pd.DataFrame({
    'path': [os.path.join('png', f'{d}.png') for d in range(10)],
    'digit': list(range(10)),
})
goals['image'] = goals['path'].apply(lambda p: imageio.imread(p))

train = labels[labels.index % 5 != 0]
valid = labels[labels.index % 5 == 0]

image_shape = train.iloc[0]['image'].shape
weights = np.zeros(image_shape, dtype=np.double)

alpha = 1e-6

for i in range(1):
    for j in range(len(train)):
        image = train.iloc[j]['image']
        goal_digit = train.iloc[j]['digit']
        goal = goals.iloc[goal_digit]['image']

        # FIXME: create one probability prediction per outcome
        prediction = image.dot(weights)
        delta = goal - prediction 

        weight_delta = delta * image
        adjustment = alpha * weight_delta
        weights += adjustment

print(weights)

for i in range(len(valid)):
    image = valid.iloc[i]['image']
    digit = valid.iloc[i]['digit']

    # TODO: predict
