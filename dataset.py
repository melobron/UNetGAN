import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
import numpy as np
import cv2


class CelebA(Dataset):
    def __init__(self, device, data_dir='../CelebA/train', transform=None, batch_size=20, img_size=128):
        super(CelebA, self).__init__()

        self.device = device

        self.img_dir = data_dir
        all_files = os.listdir(self.img_dir)
        self.length = len(all_files)

        self.transform = transform
        self.fixed_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.fixed_indices = []
        for _ in range(batch_size):
            rand_index = np.random.randint(self.length)
            self.fixed_indices.append(rand_index)

    def fixed_batch(self):
        return torch.stack([self.random_batch(index, True) for index in self.fixed_indices]).to(self.device)

    def random_batch(self, index, fixed=False):
        file = str(index+1).zfill(6) + '.jpg'
        img_path = os.path.join(self.img_dir, file)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if fixed:
            transformed = self.fixed_transform(img)
        else:
            transformed = self.transform(img)
        return transformed

    def __getitem__(self, index):
        return self.random_batch(index)

    def __len__(self):
        return self.length
