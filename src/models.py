import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in=784, dim_hidden=200, num_class=10,p_dropout=0.5):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(dim_hidden, num_class)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x,dim=1)
        return output


class CNN(nn.Module):
    def __init__(self, dim_hidden = 512, num_class=10, p_dropout=0.5):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = (5,5))
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = (5,5))
        
        self.fc1 = nn.Linear(1024, dim_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_hidden,num_class)

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = torch.flatten(x,1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output