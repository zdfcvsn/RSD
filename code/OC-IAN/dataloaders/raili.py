import numpy as np
import cv2
import os
import random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RailiDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        mean = [0.48897059, 0.46548275, 0.4294]
        std = [0.22861765, 0.22948039, 0.24054667]
        self.normalize = transforms.Normalize(mean, std)

        cv2.setNumThreads(0)

    def _set_files(self):
        if self.split in ["train", "test"]:
            self.image_dir = os.path.join(self.root, self.split, 'images')
            self.label_dir = os.path.join(self.root, self.split, 'GroundTruth')
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        else: raise ValueError(f"Invalid split name {self.split}")

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.jpg')
        image = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
        label[label>=130] = 255
        label[label< 130] = 0
        label = label / 255
        # print('rail: {}<{}<{}'.format(image, label, image_id))
        return image, label, image_id

    def _augmentation(self, image, label):
        h, w = 1282, 160
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        # print(image.shape)
        image = image.reshape(image.shape[0],image.shape[1],-1)
        # print(image.shape)
        image = self.normalize(self.to_tensor(image))
        label = label[:,0]

        # print(label.shape)
        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        image, label = self._augmentation(image, label)
        return image, label

