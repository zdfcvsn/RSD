import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloaders import RailiDataset
from util import acc, DiceLoss
from models import OC_IAN
import numpy as np
import os

dataset_index = 'Type-I'
epoch = 80
batch_size = 5
visual_field = 40



net = OC_IAN(filed=visual_field)

net.cuda()
model_name = net.modelName()
print('{} download successful'.format(model_name))

dir_checkpoint = './save/checkpoints/' + dataset_index + '/' + model_name + '/'
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)

train_set = RailiDataset(root='./../../data/RSDDs/Type-I', split='train')
train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20, 30, 40, 60, 80], gamma=0.5)

loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1,50]).cuda())
# loss_func = DiceLoss()

avg_acc = 0
best_acc = 0
low_loss = 100

net.train()
for e_poch in range(epoch):
    avg_acc = 0
    avg_loss = 0
    step = 0
    for _, (x, y) in enumerate(train_data):
        x = x.cuda()
        y = y.cuda().long()
        # print(y.size(1))
        for start_line in range(visual_field,y.size(1), int(visual_field/2)):
            step = step + 1
            x1 = x[:,:,start_line-visual_field:start_line,:]
            y1 = y[:, start_line-1]
            # print(y1.size())
            # x1 = x1.view(-1,80,160)
            y1 = y1.view(-1)
            output = net(x1)
            # print(output.size())
            loss = loss_func(output, y1)
            train_acc = acc(output, y1)
            avg_loss = avg_loss + loss.item()
            avg_acc = avg_acc + train_acc.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item())
    lr_schedule.step()
    avg_acc = avg_acc/step
    avg_loss = avg_loss/step
    print('epoch-{}, avg_loss:{}, avg_acc:{}'.format(e_poch, avg_loss, avg_acc))

    if avg_acc > best_acc:
        if low_loss > avg_loss:
            torch.save(net.state_dict(), dir_checkpoint + 'rail_acc.pth')
            low_loss = avg_loss
            best_acc = avg_acc


