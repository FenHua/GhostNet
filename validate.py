import os
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from ghostnet import ghostnet
from collections import OrderedDict
import torchvision.datasets as datasets
import torchvision.transforms as transforms
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--data', metavar='DIR', default='/cache/data/imagenet/',help='path to dataset')        # 验证数据的位置
parser.add_argument('--output_dir', metavar='DIR', default='/cache/models/',help='path to output files')    # 输出文件夹
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size')        # BS
parser.add_argument('--num-classes', type=int, default=1000,help='Number classes in dataset')               # 类别数
parser.add_argument('--width', type=float, default=1.0, help='Width ratio (default: 1.0)')     # 模型宽度，控制每层kernel数量
parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',help='Dropout rate (default: 0.2)')
parser.add_argument('--num-gpu', type=int, default=1, help='Number of GPUS to use')                         # 可以使用的gpu数


def main():
    args = parser.parse_args()                                                               # 输入解析
    model = ghostnet(num_classes=args.num_classes, width=args.width, dropout=args.dropout)   # 构建模型
    model.load_state_dict(torch.load('./models/state_dict_93.98.pth'))                       # 检查点恢复
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()    # 并行操作
    elif args.num_gpu < 1:
        model = model
    else:
        model = model.cuda()
    print('GhostNet created.')
    valdir = os.path.join(args.data, 'val')                                                  # 数据验证集
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    loader = torch.utils.data.DataLoader( datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])),
                                          batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)
    model.eval()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()                                         # 验证集损失函数
    eval_metrics = validate(model, loader, validate_loss_fn, args)
    print(eval_metrics)                                                                     # 度量结果


# 验证函数
def validate(model, loader, loss_fn, args, log_suffix=''):
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
                log_name = 'Test' + log_suffix
                logging.info(
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


if __name__ == '__main__':
    main()
