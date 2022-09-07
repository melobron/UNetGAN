import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch import autograd
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F

from models import Generator, UNet_Discriminator
from dataset import CelebA
from utils import *

import time
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm


class TrainUNetGAN:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.dim_z = args.dim_z

        self.G_lr = args.G_lr
        self.D_lr = args.D_lr
        self.B1 = args.B1  # 0.0
        self.B2 = args.B2  # 0.999

        # Transformation Parameters

        # Models
        self.G = Generator().to(self.device)
        self.D = UNet_Discriminator().to(self.device)

        # Optimizer
        self.G_optim = optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(self.B1, self.B2))
        self.D_optim = optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(self.B1, self.B2))

        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.patch_size),
            transforms.CenterCrop(self.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Dataset
        self.dataset = CelebA(device=self.device, transform=transform, batch_size=self.batch_size, img_size=self.patch_size)
        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True, drop_last=True)

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs):
            for i, img in enumerate(self.dataloader):
                self.G.train()
                self.D.train()
                self.G_optim.zero_grad()
                self.D_optim.zero_grad()

                img = img.to(self.device)
                img.requires_grad = False

                # Train Discriminator
                toggle_grad(self.D, True)
                toggle_grad(self.G, False)

                self.D_optim.zero_grad()
                D_real_target = torch.tensor([1.0]).to(self.device)
                D_fake_target = torch.tensor([0.0]).to(self.device)

                z = torch.randn(self.batch_size, self.dim_z, requires_grad=False).to(self.device)

                fake_img = self.G(z)
                D_fake, D_middle_fake = self.D(fake_img)
                D_real, D_middle_real = self.D(img)

                D_loss_real_2d = F.binary_cross_entropy_with_logits(D_real.view(-1), D_real_target.expand_as(D_real.view(-1)))
                D_loss_fake_2d = F.binary_cross_entropy_with_logits(D_fake.view(-1), D_fake_target.expand_as(D_fake.view(-1)))

                D_loss_real_middle = F.binary_cross_entropy_with_logits(D_middle_real.view(-1), D_real_target.expand_as(D_middle_real.view(-1)))
                D_loss_fake_middle = F.binary_cross_entropy_with_logits(D_middle_fake.view(-1), D_fake_target.expand_as(D_middle_fake.view(-1)))

                D_loss_real = D_loss_real_2d + D_loss_real_middle
                D_loss_fake = D_loss_fake_2d + D_loss_fake_middle

                D_loss = 0.5 * (D_loss_real + D_loss_fake)

                D_loss.backward()
                self.D_optim.step()
                del D_loss

                # Train Generator
                toggle_grad(self.D, False)
                toggle_grad(self.G, True)

                self.G_optim.zero_grad()

                z = torch.randn(self.batch_size, self.dim_z, requires_grad=False).to(self.device)
                G_fake_target = torch.tensor([1.0]).to(self.device)

                fake_img = self.G(z)
                G_fake, G_middle_fake = self.D(fake_img)

                G_loss_fake_2d = F.binary_cross_entropy_with_logits(G_fake.view(-1), G_fake_target.expand_as(G_fake.view(-1)))
                G_loss_fake_middle = F.binary_cross_entropy_with_logits(G_middle_fake.view(-1), G_fake_target.expand_as(G_middle_fake.view(-1)))

                G_loss = 0.5 * (G_loss_fake_2d + G_loss_fake_middle)

                G_loss.backward()
                self.G_optim.step()
                del G_loss

                # Loss items
                D_loss_real_2d_item = D_loss_real_2d.item()
                D_loss_real_middle_item = D_loss_real_middle.item()
                D_loss_fake_2d_item = D_loss_fake_2d.item()
                D_loss_fake_middle_item = D_loss_fake_middle.item()
                D_loss_item = D_loss_real_2d_item + D_loss_real_middle_item + D_loss_fake_2d_item + D_loss_fake_middle_item

                G_loss_fake_2d_item = G_loss_fake_2d.item()
                G_loss_fake_middle_item = G_loss_fake_middle.item()
                G_loss_item = G_loss_fake_2d_item + G_loss_fake_middle_item

                print('Epoch:{}/{}, Data:{}/{} | D_total:{:.3f}, G_total:{:.3f} | '
                      'D_real:{:.3f}, D_real_m:{:.3f}, D_fake:{:.3f}, D_fake_m:{:.3f}, G_fake:{:.3f}, G_fake_m:{:.3f}'.format(
                    epoch, self.n_epochs, self.batch_size * (i+1), len(self.dataset), D_loss_item, G_loss_item,
                    D_loss_real_2d_item, D_loss_real_middle_item, D_loss_fake_2d_item, D_loss_fake_middle_item,
                    G_loss_fake_2d_item, G_loss_fake_middle_item
                ))

                # Save Evaluated Images
                with torch.no_grad():
                    z = torch.randn(1, self.dim_z, requires_grad=False).to(self.device)

                    self.G.eval()
                    sample_fake = self.G(z)
                    sample_fake = torch.squeeze(sample_fake, dim=0).cpu()
                    sample_fake = 0.5*(sample_fake+1.0)
                    sample_fake = torch.clamp(sample_fake, 0., 1.)

                # Summary Writer
                self.summary.add_scalar('D_Loss', D_loss_item, epoch)
                self.summary.add_scalar('G_Loss', G_loss_item, epoch)
                self.summary.add_scalar('Total_loss', D_loss_item + G_loss_item, epoch)
                self.summary.add_image('Evaluated_image', sample_fake, epoch, dataformats='CHW')

            # Checkpoints
            if epoch % 250 == 0 or epoch == self.n_epochs:
                torch.save(self.G.state_dict(), os.path.join(self.checkpoint_dir, '{}epochs.pth'.format(epoch)))

        self.summary.close()
