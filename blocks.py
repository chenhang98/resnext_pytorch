import torch
import torch.nn as nn
import torch.nn.functional as F


class resBlock(nn.Module):
    def __init__(self, input_channel, factor = 1, downsample = False, D = 64, C = 8):
        super(resBlock, self).__init__()

        # channel size for 3 Convs
        c1, c2, c3 = C*D*factor, C*D*factor, 4*D*factor
        self.stride = 2 if downsample else 1
        self.shortcut = (input_channel != c3 or downsample)

        # for bottleneck:
        # 1x1, 64   conv
        self.conv1 = nn.Conv2d(input_channel, c1, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c1)
        # 3x3, 64   conv
        self.conv2 = nn.Conv2d(c1, c2, kernel_size = 3, stride = self.stride, padding = 1, groups = C, bias = False)
        self.bn2 = nn.BatchNorm2d(c2)
        # 1x1, 256  conv
        self.conv3 = nn.Conv2d(c2, c3, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(c3)

        # for shortcut
        if self.shortcut:
            self.conv0 = nn.Conv2d(input_channel, c3, kernel_size = 1, stride = self.stride, bias = False)
            self.bn0 = nn.BatchNorm2d(c3)


    def forward(self, inputs):
        # bottleneck path
        x = F.relu(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        sc = self.bn0(self.conv0(inputs)) if self.shortcut else inputs

        if x.shape != sc.shape:
            print(x.shape, sc.shape)
        # merge two paths
        assert x.shape == sc.shape, "merge failed in resBlock"
        return F.relu(x + sc, inplace = True)


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)

    print(x.shape)
    b = resBlock(3, 1)
    y = b(x)
    print(b)
    print(y.shape, y.max(), y.min())