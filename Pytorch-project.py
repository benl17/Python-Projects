import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(training = True):
    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if(training == True):
        train_set=datasets.FashionMNIST('./data',train=True,
            download=True,transform=custom_transform)
        return torch.utils.data.DataLoader(train_set, batch_size = 64)

    elif(training == False):
        test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)
        return torch.utils.data.DataLoader(test_set, batch_size = 64)
        
def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for currEpoch in range(T):
        correct = 0
        lossTotal = 0
        total = 0
        for currData in train_loader:
            inputs, labels = currData
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            nothing, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            lossTotal += loss.item()*64
        print("Train Epoch: {} Accuracy: {}/{} ({:.2f}%) Loss: {:.3f}".format(currEpoch, correct, total, (correct/total * 100), (lossTotal/total)))
    
def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    with torch.no_grad():
        correct = 0
        lossTotal = 0
        total = 0
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            nothing, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            lossTotal += loss.item()*data.shape[0]
            
    if(show_loss == True):
        print("Average loss: {:.4f}".format(round(lossTotal/total, 4)))
    print("Accuracy: {:.2f}%".format(round(correct/total * 100, 2)))

def predict_label(model, test_images, index):
    logits = model(test_images)
    prob = F.softmax(logits, dim=1)
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt'
                ,'Sneaker','Bag','Ankle Boot']
    currProbs = prob[index].tolist()
    outputList = list()
    
    for i in range(len(currProbs)):
        outputList.append([class_names[i],float(currProbs[i]) * 100])
    outputList = sorted(outputList, key = lambda x: x[1], reverse = True)[:3]
    
    for currItem in outputList:
        print("{}: {:.2f}%".format(currItem[0], currItem[1]))