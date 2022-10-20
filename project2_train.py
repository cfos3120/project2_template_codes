'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

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

# parser = argparse.ArgumentParser(description= \
#                                      'scipt for training of project 2')
# parser.add_argument('--cuda', action='store_true', default=False,
#                     help='Used when there are cuda installed.')
# args = parser.parse_args()

# training process. 
def train_net(net, trainloader, valloader):
########## ToDo: Your codes goes below #######
    
    # Training Settings
    epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # Initialise lists for storing statistics
    global device
    val_accuracy = 0
    train_loss_list = list()
    val_acc_list = list()
    # val_accuracy is the validation accuracy of each epoch. You can save your model base on the best validation accuracy.
    for epoch in range(epochs):

        for iter, data in enumerate(trainloader,0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 2000 == 1999:    # print every 2000 mini-batches
                print('Epoch {}, Iter {}, Loss {:.2f}'.format(epoch, iter, loss))
        
        # store loss log
        train_loss_list.append(loss.item())

        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # store loss log
        accuracy = correct / total
        val_acc_list.append(accuracy)

        # print epoch output stat
        print('Training Loss {:.4f} and Validation Accuracy {:.2f}'.format(loss.item(),accuracy))
        
        # save model
        torch.save(net.state_dict(), 'model.pth')
    return val_accuracy

##############################################

############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.
if __name__ == '__main__':

    train_transform = transforms.Compose([
        #transforms.RandomCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ####################################

    ####################################
    # Define the training dataset and dataloader.
    # You can make some modifications, e.g. batch_size, adding other hyperparameters, etc.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_path = '/project/MLFluids/5307Project2'
    #image_path = r'Z:\PRJ-MLFluids\elec\5307Project2'
    imageset = ImageFolder(image_path, train_transform)

    imageset_length = len(imageset)
    imageset_i = list(range(imageset_length))

    split_i = int(0.2 * imageset_length)

    training_i, validation_i = imageset_i[split_i:], imageset_i[:split_i]

    train_sampler = torch.utils.data.SubsetRandomSampler(training_i)
    val_sampler = torch.utils.data.SubsetRandomSampler(validation_i)


    # DELETE THIS later when you have the full dataset
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=train_transform)
    # valset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                         download=True, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(imageset, batch_size=4,
                                            shuffle=False, num_workers=2, sampler = train_sampler)
    valloader = torch.utils.data.DataLoader(imageset, batch_size=4,
                                            shuffle=False, num_workers=2, sampler = val_sampler)
    ####################################

    # ==================================
    # use cuda if called with '--cuda'.

    network = Network(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072
        )

    network.to(device)
    # if args.cuda:
    #     network = network.cuda()

    # train and eval your trained network
    # you have to define your own 
    val_acc = train_net(network, trainloader, valloader)

    print("final validation accuracy:", val_acc)

    # ==================================
