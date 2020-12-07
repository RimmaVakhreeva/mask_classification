import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

__all__ = ['FaceMaskDataset']


class FaceMaskDataset(Dataset):
    def __init__(self, images_data, transform):
        self.images_data = images_data
        self.transform = transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        img_filename, label = self.images_data[idx]
        image = cv2.imread(str(img_filename))[..., ::-1]
        image = self.transform(image=image)['image']
        return {'image': image, 'label': torch.from_numpy(np.array([label], dtype=np.long))}
