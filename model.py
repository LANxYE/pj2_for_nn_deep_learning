# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 第一组卷积 + BN + ReLU + 池化
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二组卷积 + BN + ReLU + 池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三组卷积 + BN + ReLU
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(8 * 8 * 128, 256)  # 假设输入32x32，经过两次2x2池化 => 8x8
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 16, 16]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 8, 8]
        x = F.relu(self.bn3(self.conv3(x)))              # [B, 128, 8, 8]
        x = x.view(x.size(0), -1)                         # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
