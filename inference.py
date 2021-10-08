'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    Compose, ShiftScaleRotate, Blur, Resize, Cutout
)

import timm
from sklearn.metrics import roc_auc_score

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils import progress_bar
from dataset import SETIDataset

parser = argparse.ArgumentParser(description='PyTorch SETI test')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--img', default=224, type=int, help='image size')
parser.add_argument('--epoch',default=80, type=int, help='num of epoch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_score = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = albumentations.Compose([
    Resize(args.img,args.img),
    ToTensorV2()
])

path=Path('/home/datasets/SETI')
df_test=pd.read_csv('./sample_submission.csv')
testset=SETIDataset(df=df_test, path=path, transform=transform_test, is_train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=4)


# Model
print('==> Building model..')
net=timm.create_model('nfnet_l0',pretrained=True,in_chans=1,num_classes=1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


checkpoint=torch.load('./runs/Aug17_14-21-38/last_ckpt.pth')
net.load_state_dict(checkpoint)


def test():
    net.eval()
    all_predictions=[]

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            all_predictions.extend(outputs.cpu().detach().numpy().tolist())
    
            progress_bar(batch_idx, len(testloader))
    return all_predictions


predictions=test()
predictions=np.array(predictions)
predictions=(predictions-predictions.min())/(predictions.max()-predictions.min())
df_test.target=predictions
df_test.to_csv('submission_nfnet_l0_last_640_withoutgm.csv',index=False)