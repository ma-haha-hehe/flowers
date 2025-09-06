import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

#数据读取与预处理操作
data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

#制作数据源
#data_transforms指定所有图像预处理操作
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),#随即旋转
        transforms.CenterCrop(224),#中心裁剪 224*224
        transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转
        transforms.RandomVerticalFlip(p=0.5), #随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomGrayscale(p=0.025),#转为灰度图
        transforms.ToTensor(), #转换成tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



batch_size = 4  #一次从数据集中取4张图片组成一个batch送进模型
#构建数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
#batch单位取数据
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
#数据数量
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


#名字字符串 读取标签对应的实际名字
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)




#加载models中提供的模型，直接用训练好的权重当作初始化参数
model_name = 'resnet'
feature_extract = True #是否用人家训练好的特征来做
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #用GPU/CPU来跑


#冻结/解冻数据
#冻结所有参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False      #冻结所有  除非后续手动修改某些层的requires_grad



#选择合适的模型，不同模型的初始化方法稍微有点区别
#初始化这个模型 加载预训练ResNet152，并且修改全连接层

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True): #use_pretrained=True说明加载与训练模型
    model_ft = None
    input_size = 0
    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained) #模型加载进来
        set_parameter_requires_grad(model_ft, feature_extract) #冻住所有层 feature_extract=True
        num_ftrs = model_ft.fc.in_features                     #获取ResNet152全连接层（fc）的输入特征数=num_ftrs
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102), #替换全连接层 102代表新的分类数 fc层被重新定义 默认requires_grad=True 是可训练的
                                   nn.LogSoftmax(dim=1))                 #在维度上做log softmax
        input_size = 224

        return model_ft, input_size




#设置哪些层需要训练 调用initialize_model
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
#GPU计算
model_ft = model_ft.to(device) #将模型移动到指定的计算设备
#模型保存
filename='checkpoint.pth'


# 是否训练所有层（只是打印）
params_to_update = model_ft.parameters()
print("Params to learn:") #需要训练的层
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)



# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()




#训练模块
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()                            #记录当前时间 用于之后计算训练耗时
    best_acc = 0                                   #保存最好的模型 初始化最佳验证准确率为0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)                               #将模型加载到指定的设备

    val_acc_history = []                           #每轮epoch结束后验证集的准确率历史
    train_acc_history = []
    train_losses = []                              #每轮的训练损失
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]        #每轮训练后优化器的学习率变化

    best_model_wts = copy.deepcopy(model.state_dict())   #最好的model的参数

    for epoch in range(num_epochs):                      #遍历每一个epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        #训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍 取数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)   #得到预测结果
                        loss = criterion(outputs, labels)   #计算损失

                    _, preds = torch.max(outputs, 1)  #概率最大的值=preds

                    # 训练阶段更新权重 梯度下降更新参数
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



            # 得到最好那次的模型 模型保存
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc  #最好的一次
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  #模型参数
                    'best_acc': best_acc,              #效果
                    'optimizer': optimizer.state_dict(), #优化器
                }
                torch.save(state, filename)  #保存下来 保存到filename路径下
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)  #准确率最高的那一次的model
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs



#开始训练

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=20, is_inception=(model_name=="inception"))
