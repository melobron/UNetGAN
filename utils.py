import os
import math
import random

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler


################################# Path & Directory #################################
def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def make_exp_dir(main_dir):
    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


################################# Model #################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


################################# Transforms #################################
def get_transforms(args):
    transform_list = []
    if args.flip_rotate:
        transform_list.append(A.OneOf([
            A.HorizontalFlip(p=1),
            A.RandomRotate90(p=1),
            A.VerticalFlip(p=1)
        ], p=0.5))
    if args.resize:
        transform_list.append(A.Resize(args.patch_size, args.patch_size))
    if args.normalize:
        # transform_list.append(A.Normalize(mean=args.mean, std=args.std, max_pixel_value=255.0))
        transform_list.append(A.Normalize(mean=args.mean, std=args.std, max_pixel_value=1.0))
    transform_list.append(ToTensorV2())
    return transform_list


################################# Data Processing #################################
def get_patches(image, patch_size, stride):
    patches = []
    h, w = image.shape
    for h_start in range(0, h, stride):
        for w_start in range(0, w, stride):
            h_end = h_start + patch_size
            w_end = w_start + patch_size
            if h_end <= h and w_end <= w:
                patches.append(image[h_start:h_end, w_start:w_end])
    return patches


def make_image_from_patches(image, patches, patch_size):
    full_image = np.zeros_like(image)
    h, w = image.shape
    index = 0
    for h_start in range(0, h, patch_size):
        for w_start in range(0, w, patch_size):
            h_end, w_end = h_start + patch_size, w_start + patch_size
            if h_end <= h and w_end <= w:
                full_image[h_start:h_end, w_start:w_end] = patches[index]
            index += 1
    return full_image


def mirror_pad(image, multiple):
    h, w = image.shape
    val_size = (max(h, w) + (multiple-1)) // multiple * multiple
    pad = ((0, val_size-h), (0, val_size-w))
    padded_image = np.pad(image, pad, mode='reflect')
    return padded_image


def random_crop(img, patch_h, patch_w):
    h, w = img.shape
    h_new = random.randrange(0, h-patch_h)
    w_new = random.randrange(0, w-patch_w)
    return img[h_new:h_new+patch_h, w_new:w_new+patch_w]


################################# UNetGAN Training #################################
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off







################################# ETC #################################
def tensor_to_numpy(tensor):
    tensor = torch.clamp(tensor, 0., 1.)
    img = tensor.mul(255).to(torch.uint8)
    img = img.numpy().transpose(1, 2, 0)
    return img


def denorm(tensor, mean=0.5, std=0.5, max_pixel=1.):
    return std*max_pixel*tensor + mean*max_pixel


def get_statistics(imgs, img_size=512, max_pixel=255.):
    linear_sum, square_sum = 0, 0
    for img in imgs:
        img = img / max_pixel
        linear_sum += img.sum()
        square_sum += np.square(img).sum()
    count = img_size * img_size * len(imgs)
    mean = linear_sum / count
    var = square_sum / count - (mean ** 2)
    std = np.sqrt(var)
    print('mean:{}, std:{}'.format(mean, std))
    return {'mean': mean, 'std': std}


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

