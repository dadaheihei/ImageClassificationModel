import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import argparse
from adversary import Attack
from data import CIFAR10
from resnet import resnet20

# 设置随机种子保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 参数解析
parser = argparse.ArgumentParser(description='CIFAR10对抗训练')
parser.add_argument('--batch_size', type=int, default=128, help='训练batch大小')
parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
parser.add_argument('--data_dir', type=str, default='./data', help='数据集路径')
parser.add_argument('--save_dir', type=str, default='./models', help='模型保存路径')
parser.add_argument('--use_cuda', action='store_true', default=True, help='是否使用CUDA')
args = parser.parse_args()

# 创建模型保存目录
os.makedirs(args.save_dir, exist_ok=True)

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
train_dataset = CIFAR10(root=args.data_dir, train=True, transform=transform_train, download=True)
test_dataset = CIFAR10(root=args.data_dir, train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# 初始化模型
def init_model():
    model = resnet20()
    if args.use_cuda and torch.cuda.is_available():
        model = model.cuda()
    return model

# 标准训练函数
def train_standard(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
    
    return train_loss/(batch_idx+1), 100.*correct/total

# 对抗训练函数
def train_adversarial(model, train_loader, optimizer, criterion, epoch, attack):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # 生成对抗样本
        inputs_adv, _, _ = attack.fgsm(inputs, targets, targeted=False, eps=0.03)
        
        optimizer.zero_grad()
        
        # 同时使用原始样本和对抗样本训练
        outputs = model(inputs)
        outputs_adv = model(inputs_adv)
        
        loss = criterion(outputs, targets) + criterion(outputs_adv, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
    
    return train_loss/(batch_idx+1), 100.*correct/total

# 测试函数
def test(model, test_loader, criterion, attack=None):
    model.eval()
    test_loss = 0
    correct = 0
    correct_adv = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # 测试原始样本
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 测试对抗样本（如果有）
            if attack is not None:
                inputs_adv, _, _ = attack.fgsm(inputs, targets, targeted=False, eps=0.03)
                outputs_adv = model(inputs_adv)
                _, predicted_adv = outputs_adv.max(1)
                correct_adv += predicted_adv.eq(targets).sum().item()
    
    acc = 100.*correct/total
    acc_adv = 100.*correct_adv/total if attack is not None else 0
    
    print(f'Test Loss: {test_loss/(batch_idx+1):.3f} | Acc: {acc:.2f}%', end='')
    if attack is not None:
        print(f' | Adv Acc: {acc_adv:.2f}%')
    else:
        print()
    
    return test_loss/(batch_idx+1), acc, acc_adv

# 学习率调度
def adjust_learning_rate(optimizer, epoch):
    """学习率衰减"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 主训练流程
def main():
    # 初始化标准模型
    standard_model = init_model()
    standard_criterion = nn.CrossEntropyLoss()
    standard_optimizer = optim.SGD(standard_model.parameters(), lr=args.lr, 
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 初始化对抗训练模型
    adv_model = init_model()
    adv_criterion = nn.CrossEntropyLoss()
    adv_optimizer = optim.SGD(adv_model.parameters(), lr=args.lr, 
                             momentum=args.momentum, weight_decay=args.weight_decay)
    
    # 初始化攻击器
    attack = Attack(adv_model, adv_criterion, cuda=args.use_cuda)
    
    # 训练日志
    standard_log = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_adv_acc': []}
    adv_log = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_adv_acc': []}
    
    # 训练循环
    for epoch in range(args.epochs):
        adjust_learning_rate(standard_optimizer, epoch)
        adjust_learning_rate(adv_optimizer, epoch)
        
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        # 标准模型训练
        print('\nTraining Standard Model:')
        train_loss, train_acc = train_standard(standard_model, train_loader, 
                                             standard_optimizer, standard_criterion, epoch)
        test_loss, test_acc, test_adv_acc = test(standard_model, test_loader, standard_criterion, attack)
        
        standard_log['train_loss'].append(train_loss)
        standard_log['train_acc'].append(train_acc)
        standard_log['test_loss'].append(test_loss)
        standard_log['test_acc'].append(test_acc)
        standard_log['test_adv_acc'].append(test_adv_acc)
        
        # 对抗训练模型训练
        print('\nTraining Adversarial Model:')
        train_loss, train_acc = train_adversarial(adv_model, train_loader, 
                                                adv_optimizer, adv_criterion, epoch, attack)
        test_loss, test_acc, test_adv_acc = test(adv_model, test_loader, adv_criterion, attack)
        
        adv_log['train_loss'].append(train_loss)
        adv_log['train_acc'].append(train_acc)
        adv_log['test_loss'].append(test_loss)
        adv_log['test_acc'].append(test_acc)
        adv_log['test_adv_acc'].append(test_adv_acc)
        
        # 保存模型
        if (epoch+1) % 10 == 0 or epoch == args.epochs-1:
            torch.save(standard_model.state_dict(), 
                      os.path.join(args.save_dir, f'standard_model_epoch{epoch+1}.pth'))
            torch.save(adv_model.state_dict(), 
                      os.path.join(args.save_dir, f'adversarial_model_epoch{epoch+1}.pth'))
    
    # 打印最终结果
    print('\nFinal Results:')
    print('Standard Model:')
    print(f'Test Accuracy: {standard_log["test_acc"][-1]:.2f}%')
    print(f'Adversarial Test Accuracy: {standard_log["test_adv_acc"][-1]:.2f}%')
    
    print('\nAdversarial Model:')
    print(f'Test Accuracy: {adv_log["test_acc"][-1]:.2f}%')
    print(f'Adversarial Test Accuracy: {adv_log["test_adv_acc"][-1]:.2f}%')

if __name__ == '__main__':
    main()