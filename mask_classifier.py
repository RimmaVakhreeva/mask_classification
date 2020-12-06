import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim

to_with_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/with_mask')
to_without_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/without_mask')

images_with_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/with_mask')
images_without_mask = Path('/media/svakhreev/fast/rimma_work/Face_mask_detection/without_mask')

def load_images(folder, masks_folder_name='with_mask'):
    output_data = []
    img_suffixes = {'.png', '.jpg', '.jpeg'}
    for image_filename in tqdm(folder.iterdir()):
        if image_filename.suffix.lower() not in img_suffixes:
            continue
        output_data.append((image_filename, int(image_filename.parent.name == masks_folder_name)))
    return output_data

train_images_data = []
train_images_data += load_images(images_with_mask)
train_images_data += load_images(images_without_mask)

class FaceMaskDataset(Dataset):
    def __init__(self, images_data, transform):
        self.images_data = images_data
        self.transform = transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        img_filename, label = self.images_data[idx]
        image = Image.open(str(img_filename))
        image = self.transform(image)
        return {'image': image, 'label': label}

train_dataset = FaceMaskDataset(train_images_data,
                                transform=transforms.Compose([
                                    transforms.Resize((112, 112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0, 0, 0], std=[1., 1., 1.])
                                ]))
# test_dataset = FaceMaskDataset(test_folder,
#                                transform=transforms.Compose([
#                                    transforms.Resize((112, 112)),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize(mean=[0, 0, 0], std=[1., 1., 1.])
#                                ]))


batch_size = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                          sampler=val_sampler)

def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):
    loss_history = []
    train_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train()  # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y, _) in enumerate(train_loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)
            prediction = model(x_gpu)
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

    return loss_history, train_history, val_history

def compute_accuracy(model, loader):
    model.eval()
    correct_samples = 0
    total_samples = 0
    for x, y in loader:
        prediction = model(x)
        _, indices_predicted = torch.max(prediction, 1)
        correct_samples += torch.sum(indices_predicted == y)
        total_samples += y.shape[0]
        accuracy = float(correct_samples) / total_samples

        # accuracy = torch.mean(indices_predicted == y)
    return accuracy

model = models.resnet18(pretrained=True)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD( parameters, lr=0.001, momentum=0.9)
loss_history, train_history, val_history = train_model(model, train_loader, val_loader, loss, optimizer, 2)
