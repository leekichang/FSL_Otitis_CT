import os
import torch
import models
import argparse
import numpy as np
import config as cfg
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.utils.tensorboard as tb
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    parser.add_argument('--crop', default = '112', type = str)
    args = parser.parse_args()
    return args

def train(model, optimizer, criterion, trainLoader, LOSS_TRACE):
    for idx, batch in enumerate(tqdm(trainLoader)):
        optimizer.zero_grad()
        X, Y = batch
        X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
        Y_pred = model.pair_forward(X)
        Y = Y.squeeze(-1)
        LOSS = criterion(Y_pred, Y)
        LOSS_TRACE.append(LOSS.cpu().detach().numpy())
        LOSS.backward()
        optimizer.step()
        print(LOSS)
    return LOSS_TRACE

if __name__ == '__main__':
    args = parse_args()
    print(f'train.py: {args.crop}')
    
    import dataLoader
    TB_DIR = './tensorboard/'
    model = models.Cosine_ResNet50()
    model_name = f'resnet50_cosine_sim_{args.crop}'
    TB_WRITER = tb.SummaryWriter(TB_DIR + model_name)
    optimizer = optim.Adam(model.embedding_layer.parameters(), lr = 0.001)
    criterion = nn.MSELoss()
    model_save_path = f'./checkpoints/{model_name}/'
    DM = dataLoader.DataManager(f'./datasets/dataset_crop_{args.crop}/')
    dataset = DM.Load_Dataset()
    trainLoader = DM.Load_DataLoader(dataset, 171, is_train=True)
    for idx, epoch in enumerate(tqdm(range(50))):
        LOSS_TRACE = []
        LOSS_TRACE = train(model, optimizer, criterion, trainLoader, LOSS_TRACE)
        AVG_LOSS = np.average(LOSS_TRACE)
        if not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        torch.save(model.embedding_layer.state_dict(), f'{model_save_path}{epoch+1}_{AVG_LOSS:.4f}.pth')
    TB_WRITER.close()
    
