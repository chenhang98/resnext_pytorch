from torchvision.datasets import CIFAR10
import numpy as np


def get_train_data(dataset):
    return np.array([np.array(x[0]) for x in dataset])


if __name__ == "__main__":
    trainset = CIFAR10("~/dataset/cifar10", train = True)
    trainset = get_train_data(trainset)
    data = trainset.astype(np.float32)/255

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

        print("means: {}".format(means))
        print("stdevs: {}".format(stdevs))
