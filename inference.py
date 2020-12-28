from GoodVibrations import get_melspectrogram_db, spec_to_image, Audio, audio2image, train, create_resnet34

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm
import numpy as np
import os
from glob import glob
import argparse


def main(folder):

    # cpu or gpu
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    print("Inference using ", device)

    # load pretrained modeel
    model, _ = create_resnet34()
    model = model.to(device)
    model.eval()

    # prepare test data
    fnames = glob(folder + "/*")
    data = Audio(fnames, transform=transforms.ToTensor())
    test_loader = DataLoader(data, batch_size=1)
    
    # inference
    for fname, pair in zip(fnames, test_loader):
        x, _ = pair
        x = x.to(device, dtype=torch.float32)
        pred = model(x)
        print("predictions:", fname, pred.argmax(axis=1))


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--folder', type=str, help='number of epochs')
    args = my_parser.parse_args()
    folder = args.folder

    main(folder)