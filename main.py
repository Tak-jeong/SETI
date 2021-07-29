'''Train CIFAR10 with PyTorch.'''
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
from dataset import SETIDataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--img', default=224, type=int, help='image size')
parser.add_argument('--epoch',default=100, type=int, help='num of epoch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_score = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img,args.img)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img,args.img)),
    transforms.ToTensor()
])

path=Path('/home/datasets/SETI')
csv_file=path.joinpath('train_labels.csv')
df=pd.read_csv(csv_file)
train_df,test_df=train_test_split(df,test_size=0.2)

trainset=SETIDataset(df=train_df,path=path,transform=transform_train)
testset=SETIDataset(df=test_df,path=path,transform=transform_test)
    



trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net=timm.create_model('resnet50',pretrained=True,in_chans=1,num_classes=1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    all_targets = []
    all_predictions = [] 
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        all_targets.extend(targets.cpu().detach().numpy().tolist())
        all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())
        roc_auc = roc_auc_score(all_targets, all_predictions) 
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | score: %.3f '
                     % (train_loss/(batch_idx+1), roc_auc))


def test(epoch):
    global best_score
    net.eval()
    test_loss = 0
    all_targets=[]
    all_predictions=[]

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            all_targets.extend(targets.cpu().detach().numpy().tolist())
            all_predictions.extend(outputs.sigmoid().cpu().detach().numpy().tolist())
            roc_auc = roc_auc_score(all_targets, all_predictions)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | score: %.3f '
                         % (test_loss/(batch_idx+1), roc_auc))


    # Save checkpoint.
    if roc_auc > best_score:
        print('Saving..')
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/best_ckpt.pth')
        best_score = roc_auc


for epoch in range(start_epoch, start_epoch+args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()

print('Saving..')
torch.save(net.state_dict(), './checkpoint/last_ckpt.pth')

device = torch.device("cuda")
model_new = timm.create_model('resnet50',pretrained=True,in_chans=1,num_classes=1) 
model_new.load_state_dict(torch.load('./checkpoint/best_ckpt.pth'),strict=False)
model_new.to(device)