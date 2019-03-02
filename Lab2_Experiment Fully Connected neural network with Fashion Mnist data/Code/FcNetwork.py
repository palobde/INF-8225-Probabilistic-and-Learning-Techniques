import torch.nn.functional as F
from torch import nn


class FcNetworkSigmoid(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def name(self):
        return "FcNetwork_Sigmoid_{}".format(self.n)

class FcNetworkRelu(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def name(self):
        return "FcNetwork_relu_{}".format(self.n)


class FcNetworkSigmoid2C1(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def name(self):
        return "FcNetwork2C1_Sigmoid_{}".format(self.n)

class FcNetworkRelu2C1(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def name(self):
        return "FcNetwork2C1_relu_{}".format(self.n)


class FcNetworkSigmoid2C2(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, 2 * n)
        self.fc3 = nn.Linear(2 * n, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def name(self):
        return "FcNetwork2C2_Sigmoid_{}".format(self.n)

class FcNetworkRelu2C2(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, 2 * n)
        self.fc3 = nn.Linear(2 * n, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def name(self):
        return "FcNetwork2C2_relu_{}".format(self.n)
