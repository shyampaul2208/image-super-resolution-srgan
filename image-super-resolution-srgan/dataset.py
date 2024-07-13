import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

HIGH_RES = 96
LOW_RES = HIGH_RES // 4

highres_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

lowres_transform = A.Compose([
    A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])
common_transforms = A.Compose([
    A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])


HIGH_RES = 96
LOW_RES = HIGH_RES // 4

highres_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

lowres_transform = A.Compose([
    A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])

common_transforms = A.Compose([
    A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

class MyDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.data = []
        for path, subdir, files in os.walk(self.root_dir):
            for name in files:
                if(name[-3:]=="png"):
                    self.data.append(name)

    def __len__(self):
        return len(self.data)
       

    def __getitem__(self, index):
        file_name = self.data[index]
        file_path = os.path.join(self.root_dir, file_name)
        image = np.array(Image.open(file_path))
        image = common_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]
        return low_res, high_res

