# coding: utf-8
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import csv
import sys
import time

from network import ResNeXt29


def test(model, testloader, criterion):
    # test model on testloader
    # return val_loss, val_acc
    
    model.eval()
    correct, total = 0, 0
    loss, counter = 0, 0
    
    with torch.no_grad():
        for (images, labels) in testloader:

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1
    
    return loss / counter, correct / total


if __name__ == "__main__":
    start = time.time()

    # load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform_train, train = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 4)

    testset = CIFAR10("~/dataset/cifar10", transform = transform_test, train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 32, shuffle = False, num_workers = 4)

    # write header
    with open('log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])

    device_ids = [0, 1]
    # build model and optimizer
    model = nn.DataParallel(ResNeXt29(10), device_ids = device_ids)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    # model.load_state_dict(torch.load("weights.pkl"))


    # train
    i = 0
    correct, total = 0, 0
    train_loss, counter = 0, 0

    for epoch in range(0, 300):
        if epoch == 0:
            optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 5e-4, momentum = 0.9)
        elif epoch == 150:
            optimizer = optim.SGD(model.parameters(), lr = 1e-2, weight_decay = 5e-4, momentum = 0.9)
        elif epoch == 225:
            optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 5e-4, momentum = 0.9)

        # iteration over all train data
        for data in trainloader:
            # shift to train mode
            model.train()
            
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # count acc,loss on trainset
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()        
            train_loss += loss.item()
            counter += 1

            if i % 100 == 0:
                # get acc,loss on trainset
                acc = correct / total
                train_loss /= counter
                
                # test
                val_loss, val_acc = test(model, testloader, criterion)

                print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' 
                      %(i, epoch, train_loss, val_loss, acc, val_acc))
                
                # save logs and weights
                with open('log.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, train_loss, val_loss, acc, val_acc])
                torch.save(model.state_dict(), 'weights.pkl')
                    
                # reset counters
                correct, total = 0, 0
                train_loss, counter = 0, 0

            i += 1

    end = time.time()
    print("total time %.1f h" %(end - start)/3600)

