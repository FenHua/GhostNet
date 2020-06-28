# 完成不同loss函数的测试
import torch
import torch.nn as nn
'''
#BCELoss
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3,requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(m(input),target)
output.backward()
print(input)
print(target)
print(output)
print(torch.tensor(1,dtype=torch.float32))
'''

'''
# BCEwithLogitsLoss
loss = nn.BCEWithLogitsLoss()
input = torch.randn(3,requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(input,target)
print(input)
print(target)
print(output)
'''

'''
# NLLLoss
# 2D loss example
N, C = 5,4
loss = nn.NLLLoss()
# input is of size N*channel*height*width
data = torch.randn(N, 16, 10, 10)
conv = nn.Conv2d(16, C, (3,3))  # 输出为 5*8*8
m = nn.LogSoftmax(dim=1)
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0,C)
output = loss(m(conv(data)),target)
print(output)
'''

'''
# CrossEntropyLoss
loss = nn.CrossEntropyLoss()
# input is of size N*C = 3*5
input = torch.randn(3,5,requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input,target)
print(output)
'''

'''
# L1Loss
loss = nn.L1Loss()
input = torch.randn(1,2,requires_grad=True)
target = torch.randn(1,2)
output = loss(input,target)
print(output)
'''

# MSELoss(L2 norm)
loss = nn.MSELoss()
input = torch.randn(1,2,requires_grad=True)
target = torch.randn(1,2)
output = loss(input,target)
print(output)

# SmoothL1Loss
# 对异常值敏感度较低，在某些情况下，可以防止梯度爆炸式增长