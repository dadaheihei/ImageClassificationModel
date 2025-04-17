"""adversary.py"""

# 导入路径模块
from pathlib import Path

# 导入 PyTorch 相关库
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# 工具函数，如无则用户需自定义

def cuda(x, use_cuda=True):
        return x.cuda() if use_cuda and torch.cuda.is_available() else x

def where(cond, x, y):
        return torch.where(cond, x, y)

# 攻击类定义
class Attack(object):
    def __init__(self, net, criterion, cuda=False, data_loader=None, visdom=False, vf=None):
        self.net = net                        # 要攻击的模型
        self.criterion = criterion            # 损失函数
        self.cuda = cuda                      # 是否使用 CUDA
        self.data_loader = data_loader or {}  # 数据加载器
        self.visdom = visdom                  # 是否可视化
        self.vf = vf                          # 可视化函数工具
        self.eps = 1e-10                      # 防止 log(0)

    # FGSM 攻击
    def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)  # 克隆输入并设置梯度
        h_adv = self.net(x_adv)                       # 前向传播

        # 构建攻击损失（目标攻击与非目标攻击）
        cost = self.criterion(h_adv, y) if targeted else -self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)  # 清空梯度
        cost.backward()              # 反向传播

        x_adv.grad.sign_()           # 使用梯度符号
        x_adv = x_adv - eps * x_adv.grad  # 按照 FGSM 更新
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)  # 限制对抗样本范围

        h = self.net(x)              # 原始输入结果
        h_adv = self.net(x_adv)     # 对抗输入结果

        return x_adv, h_adv, h

    # I-FGSM 攻击（迭代版 FGSM）
    def i_fgsm(self, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            cost = self.criterion(h_adv, y) if targeted else -self.criterion(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha * x_adv.grad
            x_adv = where(x_adv > x + eps, x + eps, x_adv)
            x_adv = where(x_adv < x - eps, x - eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

    # 通用对抗扰动（Universal Perturbation）示例框架
    def universal(self, args):
        self.set_mode('eval')  # 设置模型为评估模式

        init = False  # 是否初始化扰动 r
        correct = 0
        cost = 0
        total = 0

        if 'test' not in self.data_loader:
            raise ValueError("请在初始化 Attack 时提供 data_loader={'test': dataloader} 字典")

        data_loader = self.data_loader['test']
        for e in range(100000):
            for batch_idx, (images, labels) in enumerate(data_loader):
                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))

                if not init:
                    sz = x.size()[1:]
                    r = torch.zeros(sz)
                    r = Variable(cuda(r, self.cuda), requires_grad=True)
                    init = True

                logit = self.net(x + r)
                p_ygx = F.softmax(logit, dim=1)
                H_ygx = (-p_ygx * torch.log(self.eps + p_ygx)).sum(1).mean(0)
                prediction_cost = H_ygx

                perceptual_cost = -F.mse_loss(x + r, x) - F.relu(r.norm() - 5)
                cost = prediction_cost + perceptual_cost

                self.net.zero_grad()
                if r.grad:
                    r.grad.fill_(0)
                cost.backward()

                r = r + r.grad * 1e-1
                r = Variable(cuda(r.data, self.cuda), requires_grad=True)

                prediction = logit.max(1)[1]
                correct = torch.eq(prediction, y).float().mean().item()
                if batch_idx % 100 == 0:
                    if self.visdom and self.vf:
                        self.vf.imshow_multi(x.add(r).data)
                    print(correct * 100, prediction_cost.item(), perceptual_cost.item(), r.norm().item())

        self.set_mode('train')  # 恢复训练模式

    # 设置模型模式
    def set_mode(self, mode='eval'):
        if mode == 'eval':
            self.net.eval()
        else:
            self.net.train()