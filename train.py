#!/usr/bin/env python3

import imageio
import numpy as np
import pandas as pd

labels = pd.read_csv('labels.csv')
labels['image'] = labels['path'].apply(lambda p: imageio.imread(p))

train = labels[labels.index % 5 != 0]
valid = labels[labels.index % 5 == 0]

image_shape = train.iloc[0]['image'].shape
weights = np.zeros(image_shape, dtype=np.double)

alpha = 1e-3

for i in range(1):
    for j in range(len(train)):
        image = train.iloc[j]['image']
        goal = train.iloc[j]['digit']

        prediction = image.dot(weights)
