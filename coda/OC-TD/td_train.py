import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders import RailiDataset
from util import acc
from models import OC_TD
import numpy as np
import os

dataset_index = 'Type-II'

epoch = 80
batch_size = 5
visual_field = 40
#
net = OC_TD(filed=visual_field)

net.cuda()
model_name = net.modelName()
print('{} download successful'.format(model_name))

dir_checkpoint = './save/checkpoints/' + dataset_index + '/' + model_name + '/'
if not os.path.exists(dir_checkpoint):
    os.makedirs(dir_checkpoint)

train_set = RailiDataset(root='./../../data/RSDDs/Type-II', split='train')
train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10, 20, 30, 40, 60, 80, 100, 120, 130], gamma=0.5)

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
        for start_line in range(visual_field,y.size(1)-visual_field, int(visual_field/2)):
            step = step + 1
            x1 = x[:,:,start_line-visual_field:start_line,:]
            # print(x1.size())
            x2 = x[:,:,start_line+visual_field-2,:].view(x.size(0),x.size(1),-1,x.size(3))
            for i in range(visual_field-1):
                x2 = torch.cat((x2,x[:,:,start_line+visual_field-3-i,:].view(x.size(0),x.size(1),-1,x.size(3))), dim=2)
            # print(x2.size())
            y1 = y[:, start_line-1]
            y1 = y1.view(-1)
            output = net(x1,x2)
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


