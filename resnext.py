import os
import time
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Initialize program constants
USE_CUDA = False
# Check if Cuda is available
if torch.cuda.is_available():
    USE_CUDA = True
    torch.cuda.empty_cache()

NUM_EPOCHS = 50
expansion_rate = 1
kwargs = {'num_workers': 4, 'pin_memory':False} if USE_CUDA else {}
device = torch.device("cuda:0" if USE_CUDA else "cpu")

######################################################################################################

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

######################################################################################################

trainSet = torchvision.datasets.ImageFolder(root=os.path.join(os.getcwd(),
                               "train_data","medium"),transform=transforms.ToTensor())
valSet = torchvision.datasets.ImageFolder(root=os.path.join(os.getcwd(),
                               "validation_classification","medium"),transform=transforms.ToTensor())
testSet = torchvision.datasets.ImageFolder(root=os.path.join(os.getcwd(),
                               "test_classification"),transform=transforms.ToTensor())

trainLoader = DataLoader(trainSet, batch_size=128, shuffle=True, **kwargs)
valLoader = DataLoader(valSet, batch_size=128, shuffle=False, **kwargs)
testLoader = DataLoader(testSet, batch_size=64, shuffle=False, **kwargs)

NUM_CLASSES = 2300
NUM_CHANNELS = 3

nextModel = ResNet(Bottleneck, [3,4,6,3], num_classes=NUM_CLASSES,groups=32,width_per_group=4)

if USE_CUDA:
    if torch.cuda.device_count() > 1:
        nextModel = nn.DataParallel(nextModel)
    
    nextModel.to(device)

alpha = 1
criterion_label = nn.CrossEntropyLoss()
optimizer_label = torch.optim.SGD(nextModel.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, mode='min', factor=0.5, patience=0, threshold=0.001)

if __name__=="__main__":
    # Loss function and optimizers
    #optimizers = torch.optim.Adam()
    for epoch in range(NUM_EPOCHS):
        # Initialize training loop
        nextModel.train()
        avg_loss = 0.0
        for batch_num, (train, labels) in enumerate(trainLoader):
            if USE_CUDA:
                train, labels = train.cuda(), labels.cuda()

            feature, pred = nextModel(train)

            # Define the loss functions
            loss = criterion_label(pred, labels.long())

            # Update parameters for softmax loss
            optimizer_label.zero_grad()
            loss.backward() 
            optimizer_label.step()

            avg_loss += loss.item() 
            finalLoss = avg_loss
            
            if batch_num % 50 == 49:
                print("Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}".format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0  
            
            torch.cuda.empty_cache()
            del train
            del labels
            del loss

        #print(epoch+1, "Loss: ", avg_loss/len(loadTrain.dataset))
        
        # Save the model
        print(os.getcwd())
        torch.save({
                     'epoch': epoch,
                     'model_state_dict': nextModel.state_dict(),
                     'optimizer_state_dict': optimizer_label.state_dict(),
                     'loss': finalLoss,
                    }, os.path.join(os.getcwd(),"model{}.pth".format(epoch)))
        
        # Validate the model
        nextModel.eval()
        test_loss = []
        accuracy = 0.0
        total = 0

        for batch_num, (dev, labels) in enumerate(valLoader):
            if USE_CUDA:
                dev, labels = dev.cuda(), labels.cuda()

            feature, test_pred = nextModel(dev)
            _, pred = torch.max(F.softmax(test_pred, dim=1), 1)
            pred = pred.view(-1)

            loss = criterion_label(test_pred, labels.long())

            accuracy += torch.sum(torch.eq(pred, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*dev.size()[0])

            del test_pred
            del labels

        print("Test Loss: {:.6f}\n".format(np.mean(test_loss)))
        print("Overall Accuracy: {:.6f}\n".format(accuracy/total))
        scheduler.step(np.mean(test_loss))
        
        ones = np.ones(4600)
        start = 0
        
        for test, labels in testLoader:
            if USE_CUDA:
                test, labels = test.cuda(), labels.cuda()

            test_pred = nextModel(test)
            # Compute the softmax and write to csv file for evaluation
            _, pred = torch.max(F.softmax(test_pred, dim=1), 1)
            pred = pred.view(-1)
            pred = pred.cpu().numpy()
            a = np.array(trainSet.classes)

            ones[start:start+test.shape[0]] = list(a[pred])
            start+=test.shape[0]

        fileDir = os.path.join("home","vsanil","workhorse3","test_gpu","data","preds","preds{}.csv".format(epoch))
        np.savetxt(fileDir, np.dstack((np.arange(ones.size),ones))[0],delimiter=',',fmt="%d,%d")

