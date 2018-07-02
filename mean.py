from torchvision.datasets import CIFAR10
import numpy as np


if __name__ == "__main__":
    trainset = CIFAR10("~/dataset/cifar10", train = True).train_data
    data = trainset.astype(np.float32)/255.

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,i,:,:].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))