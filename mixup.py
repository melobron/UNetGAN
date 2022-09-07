import torch

import matplotlib.pyplot as plt
import numpy as np


def random_boundingbox(size, lam):
    width, height = size, size

    r = np.sqrt(1. - lam)
    w = np.int(width * r)
    h = np.int(height * r)
    x = np.random.randint(width)
    y = np.random.randint(height)

    x1 = np.clip(x - w//2, 0, width)
    y1 = np.clip(y - h//2, 0, width)
    x2 = np.clip(x + w//2, 0, width)
    y2 = np.clip(y + h//2, 0, width)

    return x1, y1, x2, y2


def CutMix(img_size):
    lam = np.random.beta(1, 1)
    x1, y1, x2, y2 = random_boundingbox(img_size, lam)
    lam = 1 - ((x2-x1)*(y2-y1) / (img_size*img_size))
    map = torch.ones((img_size, img_size))
    map[x1:x2, y1:y2] = 0
    if torch.rand(1) > 0.5:
        map = 1-map
        lam = 1-lam
    return map


def CutMixDemo():
    means = 0
    fig = plt.figure()
    for i in range(10):
        img = CutMix(128)
        means += img.mean()/10
        fig.add_subplot(2, 5, i+1)
        plt.title('{:.3f}'.format(img.mean()))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    print(means)
    plt.show()


CutMixDemo()
