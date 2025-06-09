import torch
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN

def visualize_first_conv_filters(model_path):
    # 加载模型
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 获取第一层卷积层的权重 [out_channels, in_channels, kernel_h, kernel_w]
    filters = model.conv1.weight.data.clone()

    num_filters = filters.shape[0]
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_filters):
        f = filters[i]
        # 标准化为 [0, 1]
        f_min, f_max = f.min(), f.max()
        f = (f - f_min) / (f_max - f_min)
        f = f.numpy().transpose(1, 2, 0)  # [C, H, W] → [H, W, C]

        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(f)
        plt.axis('off')

    plt.suptitle("First Conv Layer Filters", fontsize=16)
    plt.tight_layout()
    plt.savefig("filter_visualization.png")
    print("Filter image saved as filter_visualization.png")

if __name__ == '__main__':
    visualize_first_conv_filters("weights/simplecnn_cifar10.pth")
