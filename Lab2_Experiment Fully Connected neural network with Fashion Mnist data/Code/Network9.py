import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
def __init__(self):
super().__init__()
self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ReLU())
self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.ReLU()
        )
self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.ReLU()
        )

self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )


self.dense1 = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        )

self.dense2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU()
        )

self.dense3 = nn.Sequential(
            nn.Linear(512, 10)
        )

def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        dense1_out = self.dense1(res)
        dense2_out = self.dense2(dense1_out)
        out = F.log_softmax(self.dense3(dense2_out), dim=1)
return out

def description(self):
return "3 Convoluational Layer"