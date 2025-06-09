pj2/
├── codes/
│   ├── VGG_BatchNorm/            # 用于完成任务2：BN优化分析
│   │   ├── data/
│   │   │   ├── loaders.py        # CIFAR-10数据加载模块（含 get_cifar_loader, get_dataloaders）
│   │   │   └── cifar-10-batches-py/  # 已解压的CIFAR-10原始数据
│   │   ├── models/
│   │   │   ├── vgg.py            # 原始VGG-A模型（无BN）
│   │   │   └── vgg_bn_models.py  # 带BatchNorm的VGG-A模型
│   │   ├── utils/nn.py           # 初始化权重方法
│   │   ├── train_vgg_comparison.py  # 训练VGG_A vs VGG_A_BatchNorm并对比准确率
│   │   └── VGG_Loss_Landscape.py   # 用于绘制BN vs NoBN的Loss Landscape图
│
├── data/                         # 用于主模型训练的数据（同为CIFAR-10）
│   └── cifar-10-batches-py/
│
├── weights/                      # 各模型训练脚本与定义
│   ├── CNN_DeepWide.py           # 更宽更深结构模型定义
│   ├── CNN_GELUActiv.py          # 使用GELU激活函数模型定义
│   ├── model.py                  # 原始SimpleCNN模型
│   ├── model_NoBN.py             # 去除BN后的模型
│   └── filter_visualization.png  # 卷积核可视化图
│
├── landscape/                    # BN可视化结果输出目录
│   ├── loss_with_bn.png          # BN模型的Loss Landscape图
│   └── loss_no_bn.png            # 无BN模型的Loss Landscape图
│
├── train.py                      # SimpleCNN训练入口
├── train_*.py                    # 各种优化策略对应的训练脚本
├── main.py                       # 项目统一入口（可选）
└── visualize_filter.py           # 卷积核可视化脚本
