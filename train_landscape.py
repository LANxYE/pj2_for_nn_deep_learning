# train_landscape.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import SimpleCNN
from model_NoBN import SimpleCNN_NoBN

# 全局设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 数据加载
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# 超参数
LEARNING_RATES = [1e-3, 2e-3, 5e-4]
NUM_EPOCHS = 10

def train_model(model_class, label):
    loss_records = []

    for lr in LEARNING_RATES:
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(trainloader)
            losses.append(avg_loss)
            print(f"[{label} | lr={lr:.4f} | epoch={epoch+1}] loss: {avg_loss:.4f}")
        loss_records.append((lr, losses))

    return loss_records

def plot_loss_bands(results, title, output_path):
    epochs = list(range(1, NUM_EPOCHS + 1))
    max_curve = []
    min_curve = []

    for i in range(NUM_EPOCHS):
        losses_at_epoch = [record[1][i] for record in results]
        max_curve.append(max(losses_at_epoch))
        min_curve.append(min(losses_at_epoch))

    plt.figure(figsize=(8, 5))
    plt.fill_between(epochs, min_curve, max_curve, color='skyblue', alpha=0.4, label='Loss range')
    plt.plot(epochs, min_curve, label='Min Loss', color='blue')
    plt.plot(epochs, max_curve, label='Max Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print("Saved:", output_path)

if __name__ == '__main__':
    os.makedirs("landscape", exist_ok=True)

    print("=== Training WITH BatchNorm ===")
    results_bn = train_model(SimpleCNN, label="WithBN")
    plot_loss_bands(results_bn, "Loss Landscape (With BN)", "landscape/loss_with_bn.png")

    print("\n=== Training WITHOUT BatchNorm ===")
    results_nobn = train_model(SimpleCNN_NoBN, label="NoBN")
    plot_loss_bands(results_nobn, "Loss Landscape (No BN)", "landscape/loss_no_bn.png")
