# import the packages
import argparse
import logging
import sys
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from network import Network # the network you used

import numpy as np

def train(dataloader, model, loss_fn, optimizer):
    global device
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

def test(dataloader, model, loss_fn):
    global device
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # IMAGE TRANSFORMS AND LOADING
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    image_path = '/project/MLFluids/5307Project2' #r'C:\Users\Noahc\Documents\USYD\PHD\9 - Courses\ELEC5307\Assignment 2\5307Project2'
    
    imageset = ImageFolder(image_path, train_transform)
    imageset_length = len(imageset)
    imageset_i = list(range(imageset_length))
    split_i = int(0.2 * imageset_length)
    training_i, validation_i = imageset_i[split_i:], imageset_i[:split_i]

    train_sampler = torch.utils.data.SubsetRandomSampler(training_i)
    val_sampler = torch.utils.data.SubsetRandomSampler(validation_i)

    trainloader = torch.utils.data.DataLoader(imageset, batch_size=4,
                                            shuffle=False, num_workers=2, sampler = train_sampler)
    valloader = torch.utils.data.DataLoader(imageset, batch_size=4,
                                            shuffle=False, num_workers=2, sampler = val_sampler)

    # CIFAR10 Dataset for comparison
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=train_transform)
    
    trainloader2 = torch.utils.data.DataLoader(imageset, batch_size=4,
                                            shuffle=False, num_workers=2, sampler = train_sampler)
    valloader2 = torch.utils.data.DataLoader(imageset, batch_size=4,
                                            shuffle=False, num_workers=2, sampler = val_sampler)
    
    # MODEL INITIALISATION 
    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    
    
    # TRAINING
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer)
        test(valloader, model, loss_fn)
    print("Fruits Done!")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader2, model, loss_fn, optimizer)
        test(valloader2, model, loss_fn)
    print("CIFAR10 Done!")