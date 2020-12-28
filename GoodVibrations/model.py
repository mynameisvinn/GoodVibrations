import torch
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim

import os


def create_resnet34():
    """Return a modified Resnet-34 for spectrograms.
    """
    n_classes = 10
    n_input_channel = 1

    model = resnet34(pretrained=True)
    model.fc = nn.Linear(512, n_classes)
    model.conv1 = nn.Conv2d(n_input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists("./weights/acoustic.pth"):
        print("Using pretrained model for inference")
        model.load_state_dict(torch.load("./weights/acoustic.pth"))
    else:
        print("Using scratch model for inference")
        
    return model, optimizer