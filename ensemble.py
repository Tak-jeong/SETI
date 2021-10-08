import numpy as np
import pandas as pd
from timm.models import efficientnet, nfnet
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import os
import timm
import argparse
from pathlib import Path
from dataset import SETIDataset
from utils import progress_bar

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Resize



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

#Model1
net1=timm.create_model('efficientnet_bo',pretrained=True,in_chans=1,num_classes=1)
net1 = net1.to(device)
if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    cudnn.benchmark = True


checkpoint1=torch.load('./runs/Aug17_14-21-38/last_ckpt.pth')
net1.load_state_dict(checkpoint)

#Model2
net2=timm.create_model('resnet18d',pretrained=True,in_chans=1,num_classes=1)
net2 = net2.to(device)
if device == 'cuda':
    net2 = torch.nn.DataParallel(net2)
    cudnn.benchmark = True


checkpoint2=torch.load('./runs/Aug17_14-21-38/last_ckpt.pth')
net.load_state_dict(checkpoint2)


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


def test1():
    net1.eval()
    all_predictions1=[]

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            all_predictions1.extend(outputs.cpu().detach().numpy().tolist())
    
            progress_bar(batch_idx, len(testloader))
    return all_predictions1

def test2():
    net.eval()
    all_predictions2=[]

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            all_predictions2.extend(outputs.cpu().detach().numpy().tolist())
    
            progress_bar(batch_idx, len(testloader))
    return all_predictions2


predictions=test()
predictions=np.array(predictions)
predictions=(predictions-predictions.min())/(predictions.max()-predictions.min())
df_test.target=predictions
df_test.to_csv('submission_nfnet_l0_last_640_withoutgm.csv',index=False)









class ensemble(nn.Module):
    def __init__(self,nfnet:timm.create_model('nfnet_l0',pretrained=True,in_chans=1,num_classes=1),
    efficient:timm.create_model('efficientnet_b0',pretrained=True,in_chans=1,num_classes=1),
    resnet:timm.create_model('resnet18d',pretrained=True,in_chans=1,num_classes=1),input):

        self.nfnet=nfnet.to('cuda')
        self.efficient=efficient.to('cuda')
        self.resnet=resnet.to('cuda')

        self.fc1=nn.Linear(input,16)

    def __getitem__(self):
        nf=nn.DataParallel(self.nfnet).load_state_dict(torch.load())
        eff=nn.DataParallel(self.efficient).load_state_dict(torch.load())
        res=nn.DataParallel(self.resnet).load_state_dict(torch.load())
        return nf,eff,res

    def forward(self,x):
        out1=self.nf(x)
        out2=self.eff(x)
        out3=self.res(x)

        out=out1+out2+out3

        x=self.fc1(out)
        return torch.softmax(x,dim=1)