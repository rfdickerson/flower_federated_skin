
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch import nn
import os
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import flwr as fl

batch_size = 32
workers = 4
lr = 0.001
momentum = 0.9
epochs = 1

DATASET_DIR = "~/Dropbox/machine-learning/dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'), transforms.Compose([
        transforms.RandomRotation(degrees=(-20,+20)),
        transforms.ToTensor(),
        normalize,
    ]))

    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, 'val'), transforms.Compose([
        transforms.RandomRotation(degrees=(-20,+20)),
        transforms.ToTensor(),
        normalize,
    ]))

    test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'), transforms.Compose([
        transforms.RandomRotation(degrees=(-20,+20)),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    return train_loader, valid_loader, test_loader

def build_model():
    """create the CNN based on efficientnet"""
    model_conv = torchvision.models.efficientnet_b0(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    fc = nn.Sequential(
        nn.Linear(in_features=1792, out_features=625),
        nn.ReLU(),
        nn.Dropout(0.3),
        # nn.BatchNorm1d(625),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=7),
    )

    model_conv.classifier = fc
    return model_conv


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.classifier.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

net = build_model().to(DEVICE)
train_loader, valid_loader, test_loader = load_data()


class SkinLesionClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_loader, epochs=1)
        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, test_loader)
        return loss, len(test_loader.dataset), {"accuracy": accuracy}


fl.client.start_numpy_client("[::]:8080", client=SkinLesionClient())
