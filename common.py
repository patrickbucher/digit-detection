import imageio
import numpy as np
import pandas as pd


def load_input(csv_path):
    labels = pd.read_csv(csv_path)

    images = labels['path'].apply(lambda p: imageio.imread(p).flatten()).to_numpy()
    inputs = np.zeros((len(images), len(images[0])))
    for i, image in enumerate(images):
        inputs[i,:] = image

    m, n = inputs.shape

    labels = labels['digit']
    K = len(set(labels.values))

    classes = list(range(K))
    labels_1hot = {l: (labels == l).astype(int).to_numpy() for l in classes}

    return normalize(inputs), labels, labels_1hot, (K, m, n)


def normalize(x):
    return (x - x.mean()) / x.std()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
