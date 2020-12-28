import torch

from tqdm import tqdm
import numpy as np
import os


def train(model, loss_fn, train_loader, val_loader, n_epochs, optimizer):

    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')

    train_losses=[]
    val_losses=[]
    
    for epoch in tqdm(range(n_epochs)):
        
        model.train()
        batch_losses=[]

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            
            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        train_losses.append(batch_losses)
        print(f'epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

        model.eval()
        batch_losses=[]
        
        trace_y = []
        trace_yhat = []
        
        for i, data in enumerate(val_loader):
            
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            
            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            
            batch_losses.append(loss.item())
        val_losses.append(batch_losses)
        
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        print(f'epoch - {epoch} Val-Loss : {np.mean(val_losses[-1])} Val-Accuracy : {accuracy}')

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "./weights/acoustic.pth")