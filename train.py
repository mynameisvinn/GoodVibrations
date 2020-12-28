from GoodVibrations import get_melspectrogram_db, spec_to_image, Audio, audio2image, train, create_resnet34

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet34
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse



def main(folder: str, n_epochs=5) -> None:
    """entry point for model training.
    """
    fnames = glob(folder + "/*")
    data = Audio(fnames, transform=transforms.ToTensor())

    train_size = int(len(data) * .80)
    val_size = len(data) - train_size

    train_set, val_set = random_split(data, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=24, shuffle=True)

    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    print("Using", device)

    model, optimizer = create_resnet34()
    loss_fn = nn.CrossEntropyLoss()

    train(model, loss_fn, train_loader, val_loader, n_epochs, optimizer)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, help='number of epochs')
    my_parser.add_argument('--folder', type=str, help='folder for train data')
    args = my_parser.parse_args()
    n_epochs = args.epochs
    folder = args.folder

    main(folder, n_epochs)