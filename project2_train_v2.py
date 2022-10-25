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

def train(dataloader, model, loss_fn, optimizer, device):
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
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
    train_count = int(0.7 * len(imageset))
    valid_count = int(len(imageset) - train_count)
    train_dataset, valid_dataset = torch.utils.data.random_split(imageset, (train_count, valid_count))

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                            shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4,
                                            shuffle=True, num_workers=2)

    # CIFAR10 Dataset for comparison
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=train_transform)
    # valset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                         download=True, transform=train_transform)
    
    epochs = 16 

    # TRAINING ALEXNET
    model = torchvision.models.alexnet(pretrained=True)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, device)
        test(valloader, model, loss_fn, device)
    print("Fruits Done!")

    torch.save(model.state_dict(), 'project2_alexnet_pre_train.pth')

    # TRAINING ALEXNET
    model = torchvision.models.alexnet(pretrained=False)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, device)
        test(valloader, model, loss_fn, device)
    print("Fruits Done!")

    torch.save(model.state_dict(), 'project2_alexnet.pth')

    # TRAINING TRANSFORMER PRETRAIN
    model = Network(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )
    model.load_state_dict(torch.load('/project/MLFluids/model_v13.pth'))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, device)
        test(valloader, model, loss_fn, device)
    print("Fruits Done!")

    torch.save(model.state_dict(), 'project2_transformer_pre_trained.pth')
    
    # TRAINING TRANSFORMER
    model = Network(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, device)
        test(valloader, model, loss_fn, device)
    print("Fruits Done!")

    torch.save(model.state_dict(), 'project2_transformer.pth')

    # TRAINING RESNET PRETRAIN
    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, device)
        test(valloader, model, loss_fn, device)
    print("Fruits Done!")

    torch.save(model.state_dict(), 'project2_resnet_pre_train.pth')

    # TRAINING RESNET
    model = torchvision.models.resnet50(pretrained=False)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, device)
        test(valloader, model, loss_fn, device)
    print("Fruits Done!")

    torch.save(model.state_dict(), 'project2_resnet.pth')