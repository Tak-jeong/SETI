import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import timm
from sklearn.metrics import roc_auc_score

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils import progress_bar
from dataset import SETIDataset, SETIDatasettest


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--img', default=224, type=int, help='image size')
parser.add_argument('--epoch',default=80, type=int, help='num of epoch')
args = parser.parse_args()



transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img,args.img)),
    transforms.ToTensor()
])

path=Path('/home/datasets/SETI')
csv_file=path.joinpath('train_labels.csv')
df_test=pd.read_csv('./sample_submission.csv')

testset=SETIDatasettest(df=df_test,path=path,transform=transform_test)

device = torch.device("cuda")
model_new = timm.create_model('resnet50',pretrained=True,in_chans=1,num_classes=1) 
model_new.load_state_dict(torch.load('./checkpoint/best_ckpt.pth'),strict=False)
model_new.to(device)


