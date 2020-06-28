import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from ghostnet import ghostnet
import torchvision.transforms as transforms
# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = ghostnet(num_classes=10, width=1.0, dropout=0.1)
#model.load_state_dict(torch.load("models//ghostnet_10.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 定义损失函数和优化函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)  # 前35学习率0.01
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# 训练100个epoch
model.train()
for epoch in range(100):
    print("\n Epoch: %d"%(epoch+1))
    sum_loss = 0.0
    correct = 0.0
    total =0.0
    for i, data in enumerate(trainloader,0):
        length = len(trainloader)
        inputs,labels = data
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()

        # forward+backward
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        # 每个epoch输出损失和正确率
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print("[epoch:%d, iter:%d] Loss: %.03F | Acc: %.3f%%"
              %(epoch+1, (i+1+epoch*length), sum_loss/(i+1), 100.*correct/total))
    scheduler.step()      # 调整学习率，放在optimizer.step后面(参考整个epoch)
    if((epoch+1)%20==0):
        torch.save(model.state_dict(),'models/ghostnet_{}.pth'.format((epoch+1)))