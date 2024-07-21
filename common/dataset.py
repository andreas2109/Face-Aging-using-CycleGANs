import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class FaceAgingDataset(Dataset):
    def __init__(self,root_old,root_young,transform=None):
        self.root_old = root_old
        self.root_young = root_young
        self.transform = transform

        self.old_images = os.listdir(root_old)
        self.young_images = os.listdir(root_young)
        self.length_dataset = max(len(self.root_young),len(self.old_images))
        self.old_len = len(self.old_images)
        self.young_len = len(self.young_images)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self,index):
        old_img = self.old_images[index % self.old_len]
        young_img = self.young_images[index % self.young_len]

        old_path = os.path.join(self.root_old,old_img)
        young_path = os.path.join(self.root_young,young_img)
        old_img = np.array(Image.open(old_path).convert("RGB"))
        young_img = np.array(Image.open(young_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image = old_img, image0 = young_img)
            old_img = augmentations['image']
            young_img = augmentations['image0']
        return old_img,young_img