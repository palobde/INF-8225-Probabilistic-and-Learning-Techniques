import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(32*14*14, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.log_softmax(self.fc1(x), dim=1)

        return x

    def description(self):
        return "1 Convoluational Layer"
