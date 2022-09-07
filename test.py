import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torchvision.transforms import transforms

from models import Generator
import cv2
from utils import *
from pathlib import Path

# Arguments
parser = argparse.ArgumentParser(description='Test UNetGAN')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--exp_num', default=4, type=int)

# Training parameters
parser.add_argument('--n_epochs', default=250, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--dim_z', default=128, type=int)

# Transformations
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--mean', type=float, default=0.5)  # 0.39125
parser.add_argument('--std', type=float, default=0.5)  # 0.23223

opt = parser.parse_args()


def generate(args):
    device = torch.device('cuda:{}'.format(args.gpu_num))

    # Model
    model = Generator().to(device)
    model.load_state_dict(torch.load('./experiments/exp{}/checkpoints/{}epochs.pth'.format(args.exp_num, args.n_epochs), map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.patch_size),
        transforms.CenterCrop(args.patch_size),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Dataset
    data_dir = os.path.join(Path(__file__).parents[1], 'SEM_data')
    noise_dir = os.path.join(data_dir, 'Hitachi/Single_smooth_patch')
    noise_paths = make_dataset(noise_dir)

    # Save Directory
    save_dir = './experiments/exp{}/results/'.format(args.exp_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    z = torch.randn(args.batch_size, args.dim_z, requires_grad=False).to(device)
    fake = model(z).cpu()
    plt.imshow(make_grid(fake, normalize=True, nrow=2).permute(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    generate(opt)
