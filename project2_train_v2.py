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

from network import Network, MlpMixer, WarmupCosineLrScheduler # the network you used

import numpy as np

def train(dataloader, model, loss_fn, optimizer, scheduler, device):
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
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if scheduler != None: scheduler.step()

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

    return correct, test_loss

def fruit_model_runner(trainloader, valloader, model, model_save_name, device, lr, milestones, gamma, epochs=16, mixer=False):

    # TRAINING ALEXNET
    print('Starting Model Training: ', model_save_name)
    model.to(device)

    # set a scheduler:
    if mixer:
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=5e-4, 
                                    weight_decay=1e-7,
                                    momentum=0.9, 
                                    nesterov=True)
        # scheduler = WarmupCosineLrScheduler(optimizer, 
        #                             epoch, 
        #                             warmup_iter=0
        #                             )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    if milestones != None & gamma != None:
        scheduler = scheduler(optimizer, milestones=milestones, gamma=gamma)
    
    loss_fn = nn.CrossEntropyLoss()
    
    accuracy_list = list()
    loss_list = list()
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(trainloader, model, loss_fn, optimizer, scheduler, device)
        accuracy, loss = test(valloader, model, loss_fn, device)
        accuracy_list.append(accuracy)
        loss_list.append(loss)

    print(model_save_name, " Done!")
    
    return accuracy_list, loss_list

    #torch.save(model.state_dict(), model_save_name)

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

    # # TRAINING ALEXNET
    # model = torchvision.models.alexnet(pretrained=True)
    # fruit_model_runner(trainloader,valloader, model, 'project2_alexnet_pre_train.pth', device, epoch=16)

    # # TRAINING ALEXNET
    # model = torchvision.models.alexnet(pretrained=False)
    # fruit_model_runner(trainloader,valloader, model, 'project2_alexnet.pth', device, epoch=16)

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
    fruit_model_runner(trainloader,valloader, model, 'project2_transformer_pre_trained2.pth', device, epoch=32)
    
    # # TRAINING TRANSFORMER
    # model = Network(
    #     image_size=224,
    #     patch_size=16,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072
    # )
    # fruit_model_runner(trainloader,valloader, model, 'project2_transformer.pth', device, epoch=16)

    # # TRAINING RESNET PRETRAIN
    # model = torchvision.models.resnet50(pretrained=True)
    # fruit_model_runner(trainloader,valloader, model, 'project2_resnet_pre_train.pth', device, epoch=16)

    # # TRAINING RESNET
    # model = torchvision.models.resnet50(pretrained=False)
    # fruit_model_runner(trainloader,valloader, model, 'project2_resnet.pth', device, epoch=16)

    # TRAINING MLP-Mixer PRETRAIN
    # model = MlpMixer()
    # model.load_from(np.load('/project/MLFluids/Mixer-B_16.npz'))
    # fruit_model_runner(trainloader,valloader, model, 'project2_MLPMixer_pre_train.pth', device, epoch=16, mixer=True)

    # # TRAINING MLP-Mixer
    # model = MlpMixer()
    # fruit_model_runner(trainloader,valloader, model, 'project2_MLPMixer.pth', device, epoch=16, mixer=True)s


    training_conditions = {'learning_rates':[0.01, 0.005, 0.001, 0.0005, 0.0001],
                            'milestones':[None, 
                                        [2,4,8],
                                        [4,6,8],
                                        [4,8,12]],
                            'gammas':[0.25,0.5,0.75,0.9]}
    
    training_results_loss = {}
    training_results_accuracy = {}
    for lr in training_conditions['learning_rates']:
        for milestones in training_conditions['milestones']:
            for gamma in training_conditions['gammas']:

                model = Network(image_size=224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072)
                model.load_state_dict(torch.load('/project/MLFluids/model_v13.pth'))
                accuracy_list, loss_list = fruit_model_runner(trainloader, valloader, model, None, device, lr, milestones, gamma, epochs=16)
                training_results_loss['ViT-LR'+str(lr)+'-MS'+str(milestones)+'-G'+str(gamma)] = loss_list
                training_results_accuracy['ViT-LR'+str(lr)+'-MS'+str(milestones)+'-G'+str(gamma)] = accuracy_list
                
                model = torchvision.models.resnet50(pretrained=True)
                accuracy_list, loss_list = fruit_model_runner(trainloader, valloader, model, None, device, lr, milestones, gamma, epochs=16)
                training_results_loss['ResNet-LR'+str(lr)+'-MS'+str(milestones)+'-G'+str(gamma)] = loss_list
                training_results_accuracy['ResNet-LR'+str(lr)+'-MS'+str(milestones)+'-G'+str(gamma)] = accuracy_list

                model = torchvision.models.alexnet(pretrained=True)
                accuracy_list, loss_list = fruit_model_runner(trainloader, valloader, model, None, device, lr, milestones, gamma, epochs=16)
                training_results_loss['AlexNet-LR'+str(lr)+'-MS'+str(milestones)+'-G'+str(gamma)] = loss_list
                training_results_accuracy['AlexNet-LR'+str(lr)+'-MS'+str(milestones)+'-G'+str(gamma)] = accuracy_list


    import csv 
    with open('output_loss.csv', 'w') as output:
        writer = csv.writer(output)
        for key, value in training_results_loss.items():
            writer.writerow([key, value])

    with open('output_accuracy.csv', 'w') as output:
        writer = csv.writer(output)
        for key, value in training_results_accuracy.items():
            writer.writerow([key, value])