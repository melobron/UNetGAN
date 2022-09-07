import argparse
from train import TrainUNetGAN

# Arguments
parser = argparse.ArgumentParser(description='Train UNetGAN')

parser.add_argument('--exp_detail', default='UNetGAN', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Dataset parameters
# parser.add_argument('--data_dir', default='../SEM_data/Hitachi/Single_smooth_patch', type=str)
# parser.add_argument('--data_dir', default='../ImageNet_1000_Gray', type=str)

# Training parameters
parser.add_argument('--n_epochs', default=500, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--dim_z', default=128, type=int)
parser.add_argument('--G_lr', default=5e-5, type=float)
parser.add_argument('--D_lr', default=2e-4, type=float)
parser.add_argument('--B1', default=0.0, type=float)
parser.add_argument('--B2', default=0.999, type=float)

parser.add_argument('--lambda_term', default=10, type=int)

# Transformations
parser.add_argument('--patch_size', type=int, default=128)

args = parser.parse_args()

# Train Nr2N
train_UNetGAN = TrainUNetGAN(args=args)
train_UNetGAN.train()
