# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        self.convolutionOne = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0)
        self.reluOne = nn.ReLU()
        self.maxPoolOne = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.convolutionTwo = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)
        self.reluTwo = nn.ReLU()
        self.maxPoolTwo = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.flatten = nn.Flatten()

        self.linearOne = nn.Linear(in_features = 400, out_features = 256)
        self.reluThree = nn.ReLU()

        self.linearTwo = nn.Linear(in_features = 256, out_features = 128)
        self.reluFour = nn.ReLU()

        self.linearThree = nn.Linear(in_features = 128, out_features = num_classes)

    def forward(self, x):
        shape_dict = {}
        
        x1 = self.reluOne(self.convolutionOne(x))
        x1 = self.maxPoolOne(x1)
        shape_dict[1] = list(x1.size())

        x2 = self.reluTwo(self.convolutionTwo(x1))
        x2 = self.maxPoolTwo(x2)
        shape_dict[2] = list(x2.size())

        x3 = self.flatten(x2)
        shape_dict[3] = list(x3.size())

        x4 = self.reluThree(self.linearOne(x3))
        shape_dict[4] = list(x4.size())

        x5 = self.reluFour(self.linearTwo(x4))
        shape_dict[5] = list(x5.size())

        x6 = self.linearThree(x5)
        shape_dict[6] = list(x6.size())  

        out = x6    
        # certain operations
        return out, shape_dict


def count_model_params():
    model = LeNet()
    model_params = 0.0

    for name, param in model.named_parameters():
        if(param.requires_grad):
            model_params += param.numel()
    
    model_params = (model_params / 1000000)
    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc