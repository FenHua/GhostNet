# GhostNet
GhostNet；train；test；cifar-10

GhostNet: More Features from Cheap Operations. CVPR 2020. [[arXiv]](https://arxiv.org/abs/1911.11907)

By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.

首先创建文件夹data，models；
将检查点文件.pth放入models文件夹（也可以自己从头训练）


训练100epoch：
训练loss大约0.17左右，正确率大约93%

测试结果：
('loss', 0.8097940278053284), ('top1', 80.25), ('top5', 98.92)]
没有文章所介绍的效果好，可能与训练技巧以及epoch数量有关
