# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import resBlock


class ResNeXt29(nn.Module):
    def __init__(self, output_shape):
        super(ResNeXt29, self).__init__()
        self.output_shape = output_shape

        # regiter first conv layer and last fc layer
        self.conv_a = nn.Conv2d(3, 64, kernel_size = 3, padding = 1, bias = False)
        self.bn_a = nn.BatchNorm2d(64)
        self.fc = nn.Linear(1024, output_shape, bias = True)
        
        # stage 1
        self.resblocks_a1 = resBlock(64, 1)
        self.resblocks_a2 = resBlock(256, 1)
        self.resblocks_a3 = resBlock(256, 1)

        # stage 2
        self.resblocks_b1 = resBlock(256, 2, downsample = True)
        self.resblocks_b2 = resBlock(512, 2)
        self.resblocks_b3 = resBlock(512, 2)

        # stage 3
        self.resblocks_c1 = resBlock(512, 4, downsample = True)
        self.resblocks_c2 = resBlock(1024, 4)
        self.resblocks_c3 = resBlock(1024, 4)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        x = F.relu(self.bn_a(self.conv_a(inputs)), inplace = True)

        # expect x size 64x32x32 here
        assert(x.shape[1:] == torch.Size([64,32,32]))


        # stage 1
        x = self.resblocks_a1(x)
        x = self.resblocks_a2(x)
        x = self.resblocks_a3(x)

        # expect x size 256x32x32 here
        assert(x.shape[1:] == torch.Size([256,32,32]))


        # stage 2
        x = self.resblocks_b1(x)
        x = self.resblocks_b2(x)
        x = self.resblocks_b3(x)

        # expect x size 512x16x16 here
        assert(x.shape[1:] == torch.Size([512,16,16]))


        # stage 3
        x = self.resblocks_c1(x)
        x = self.resblocks_c2(x)
        x = self.resblocks_c3(x)

        # expect x size 1024x8x8 here
        assert(x.shape[1:] == torch.Size([1024,8,8]))


        # global pooling and fc
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)
    net = ResNeXt29(10)
    y = net(x)

    print(x.shape)
    print(y.shape)

    size = 1
    for param in net.parameters():
        arr = np.array(param.size())
        
        s = 1
        for e in arr:
            s *= e

        size += s

    print("all parameters %.2fM" %(size/1e6) )