import torch.nn.functional as F
from torch import nn


class CNNModel1(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.n = n
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=n, kernel_size=3, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(26*26*n, 10)

    def forward(self, x):
        out = F.sigmoid(self.cnn1(x))


        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = F.log_softmax(self.fc1(out))

        return out

    def name(self):
        return "CNNModel1_{}".format(self.n)

class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(32*4*4, 10)

    def forward(self, x):
        out = F.sigmoid(self.cnn1(x))
        out = self.maxpool1(out)

        out = F.sigmoid(self.cnn2(out))

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = F.log_softmax(self.fc1(out))

        return out
