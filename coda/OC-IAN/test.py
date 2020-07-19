import torch
import torchvision.transforms as transforms
from models import OC_IAN
import os
import numpy as np
import cv2
import os.path as osp
from PIL import Image
from util import rowtomat
import itertools

dataset_index = 'Type-I'
img_dir = './../../data/RSDDs/'+dataset_index+'/test/images'
label_dir = './../../data/RSDDs/'+dataset_index+'/test/GroundTruth'
batch_size=1
visual_field = 40

net = OC_IAN(filed=visual_field)

model_name = net.modelName()
model_pth = './save/checkpoints/' + dataset_index + '/' + model_name +'/rail_acc.pth'
net.load_state_dict(torch.load(model_pth))
net.cuda()
print(model_name + " Model loaded !")

result_path = './result/'+dataset_index+'/'+net.modelName()+'/'
if not os.path.exists(result_path):
    os.makedirs(result_path)



avg_acc = 0
net.eval()
mean = [0.48897059, 0.46548275, 0.4294]
std = [0.22861765, 0.22948039, 0.24054667]
normalize = transforms.Normalize(mean, std)
to_tensor = transforms.ToTensor()

for num_test, imgs in enumerate(os.listdir(img_dir)):
    image_path = osp.join(img_dir, imgs)
    label_path = osp.join(label_dir, imgs)
    image = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32)
    label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
    label[label >= 130] = 255
    label[label < 130] = 0
    label = label / 255
    label = torch.from_numpy(label).cuda()
    image = image.reshape(image.shape[0], image.shape[1], -1)
    x = normalize(to_tensor(image))
    y = label[:, 0]
    x = x.cuda()
    y = y.cuda()
    avg_acc = 0
    avg_loss = 0
    outputs = []
    print(x.size())
    for start_line in range(visual_field,y.size(0)):
        x1 = x[:,start_line-visual_field:start_line,:]
        y1 = y[start_line]
        # print(y1.size())
        # x1 = x1.view(-1,80,160)
        y1 = y1.long()
        output = net(x1)
        _, pred_inds = output.max(dim=1)
        # print(pred_inds[0])

        outputs.append(pred_inds[0].item())
    if np.sum(outputs) != 0:
        print(1)
    line_num = 0
    for k, v in itertools.groupby(outputs):
        cont_len = len(list(v))
        label_id = k
        if cont_len < 6 and label_id == 1:
            for i in range(cont_len):
                outputs[line_num+i] = 0
        line_num = line_num + cont_len
    # print(outputs)
    pred = rowtomat(outputs,y.size(0),visual_field)
    cv2.imwrite(result_path+imgs, pred)
    # break



