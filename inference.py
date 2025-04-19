import torch
import os
import argparse
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from torch.utils.data import DataLoader
from data import CIFAR10  # 自定义的数据集类放在 data.py 中
from torchvision import transforms

# 模型映射字典
model_dict = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet1202': resnet1202,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet32',
                        choices=model_dict.keys(), help='选择 ResNet 模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='CIFAR10 数据路径')
    args = parser.parse_args()

    # 加载模型
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在！请检查路径：{args.model_path}")

    model = model_dict[args.model]().cuda()
    try:
        checkpoint = torch.load(args.model_path)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        print(f"{args.model} 模型加载成功！")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}")

    # 加载测试集（使用自定义 data 类）
    try:
        testset = CIFAR10(args.data_path, train=False)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False)
        print("数据集加载成功！（使用自定义 data 类）")
    except Exception as e:
        raise RuntimeError(f"数据集加载失败：{str(e)}")

    # 推理整个测试集，计算准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"{args.model} 在干净测试集上的准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
