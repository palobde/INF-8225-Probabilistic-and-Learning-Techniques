import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(128*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout(x)

        x = F.sigmoid(self.conv2(x))
        x = self.maxpool2(x)
        x = self.dropout(x)

        x = F.sigmoid(self.conv3(x))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)

        x = F.log_softmax(self.fc2(x), dim=1)

        return x

    def description(self):
        return "3 Convoluational Layer"
