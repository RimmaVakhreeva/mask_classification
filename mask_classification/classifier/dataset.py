from logging import warning

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

__all__ = ['FaceMaskDataset']


class FaceMaskDataset(Dataset):
    def __init__(self, images_data, transform, debug=False):
        self.images_data = images_data
        self.transform = transform
        self._debug = debug

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        img_filename, label = self.images_data[idx]
        image = cv2.imread(str(img_filename))
        if image is None:
            warning(f'bad image: {img_filename}')
            raise FileNotFoundError
        image = self.transform(image=image[..., ::-1])['image']

        if self._debug:
            debug_image = np.transpose(image.cpu().numpy() * 255, (1, 2, 0))
            debug_image = debug_image[..., ::-1].astype(np.uint8)

            cv2.putText(debug_image, str(label), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, cv2.LINE_AA)
            cv2.imshow('img', debug_image)
            cv2.waitKey(0)

        return {'image': image, 'label': torch.from_numpy(np.array([label], dtype=np.long))}
