import os
import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
from ghostnet import ghostnet
from collections import OrderedDict
import torchvision.transforms as transforms
torch.backends.cudnn.benchmark = True


# 验证函数
def validate(model, loader, loss_fn):
    batch_time_m = AverageMeter()     # 每个batch时间度量
    losses_m = AverageMeter()         # loss度量
    top1_m = AverageMeter()           # top1准确性度量
    top5_m = AverageMeter()           # top5 准确性度量
    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx    # 是否是最后一个batch
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            loss = loss_fn(output, target)       # 计算损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))   # 返回top1，top5的正确率
            reduced_loss = loss.data             # loss大小
            torch.cuda.synchronize()             # 同步操作
            losses_m.update(reduced_loss.item(), input.size(0))  # loss度量更新
            top1_m.update(acc1.item(), output.size(0))           # top1准确性度量更新
            top5_m.update(acc5.item(), output.size(0))           # top5准确性度量更新
            batch_time_m.update(time.time() - end)               # 每个batch所用时间度量更新
            end = time.time()
            if (last_batch or batch_idx % 10 == 0):
                log_name = 'Test'
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])    # 字典类型存储验证结果
    return metrics


# 度量类
class AverageMeter:
    # 计算并存储平均和累计值
    def __init__(self):
        self.reset()    # 置零

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val                     # 当前值
        self.sum += val * n                # 累加值
        self.count += n                    # 累计数量
        self.avg = self.sum / self.count   # 平均值


# 计算top-k准确性
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)                # 取结果前k个结果保存
    batch_size = target.size(0)     # batch size大小
    _, pred = output.topk(maxk, 1, True, True)    # 前k预测结果
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))   # 比较结果
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]  # top-k正确率


model = ghostnet(num_classes=10, width=1.0, dropout=0.1)   # 构建模型
model.load_state_dict(torch.load('models//ghostnet_100.pth'))                       # 检查点恢复
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

validate_loss_fn = nn.CrossEntropyLoss().cuda()                                         # 验证集损失函数
eval_metrics = validate(model, testloader, validate_loss_fn)
print(eval_metrics)                                                                     # 度量结果
