# main.py
import torch
import torchvision
import torchvision.transforms as transforms
import os

# 配置
BATCH_SIZE = 128
NUM_WORKERS = 2

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 数据增强：裁剪
    transforms.RandomHorizontalFlip(),     # 数据增强：水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),  # 归一化
                         (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# 下载并加载数据
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    print(f'Train size: {len(trainset)} | Test size: {len(testset)}')
    # 取一个batch看看 shape 是否正常
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(f'Image shape: {images.shape} | Labels: {labels[:8]}')
