import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import functools

from layers import *


########################################## Generator Architecture ##########################################
def G_arch(ch=64, attention='64'):
    arch = {}
    arch[256] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample': [True] * 6,
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])for i in range(3,9)}}

    arch[128] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [True] * 5,
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2**i: (2**i in [int(item) for item in attention.split('_')]) for i in range(3,8)}}
    return arch


########################################## Generator ##########################################
class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128, G_attn='64',
                 n_classes=1000, num_G_SVs=1, num_G_SV_itrs=1, G_shared=True, G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, norm_style='bn'):
        super(Generator, self).__init__()

        self.ch = G_ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.resolution = resolution
        self.attention = G_attn
        self.n_classes = n_classes

        self.G_shared = G_shared
        self.shared_dim = dim_z

        self.activation = G_activation
        self.norm_style = norm_style

        self.arch = G_arch(self.ch, self.attention)[resolution]

        self.lr = G_lr
        self.B1 = G_B1
        self.B2 = G_B2

        self.which_conv = functools.partial(SNConv2d, kernel_size=3, padding=1, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs)
        self.which_linear = functools.partial(SNLinear, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs)
        self.which_embedding = nn.Embedding

        self.which_bn = bn
        self.linear = self.which_linear(self.dim_z, self.arch['in_channels'][0]*(self.bottom_width ** 2))

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                GBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv, which_bn=self.which_bn,
                       activation=self.activation,
                       upsample=(functools.partial(F.interpolate, scale_factor=2) if self.arch['upsample'][index] else None))
            ]]

            if self.arch['attention'][self.arch['resolution'][index]]:
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.output_layer = nn.Sequential(bn(self.arch['out_channels'][-1]), self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], 3))

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                init.orthogonal_(module.weight)

    def forward(self, z):  # z = (10, 128)
        h = self.linear(z)  # h = (10, 16384)
        h = h.view(h.shape[0], -1, self.bottom_width, self.bottom_width)  # h = (10, 1024, 4, 4)

        block_index = 1
        for blocklist in self.blocks:
            for block in blocklist:
                h = block(h)
            block_index += 1
        return torch.tanh(self.output_layer(h))


########################################## Discriminator Architecture ##########################################
def D_unet_arch(ch=64, attention='64'):
    arch = {}
    arch[128] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 8, 4, 2, 1, 1]],
                 'downsample': [True]*5 + [False]*5,
                 'upsample': [False]*5 + [True]*5,
                 'resolution': [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')] for i in range(2, 11)}}

    arch[256] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 16, 16, 8, 4, 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True]*6 + [False]*6,
                 'upsample': [False]*6 + [True]*6,
                 'resolution': [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256],
                 'attention': {2**i: 2**i in [int(item) for item in attention.split('_')] for i in range(2,13)}}
    return arch


########################################## Discriminator ##########################################
class UNet_Discriminator(nn.Module):
    def __init__(self, D_ch=64, resolution=128, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 D_lr=2e-4, D_B1=0.0, D_B2=0.999, output_dim=1, D_init='ortho', D_param='SN', **kwargs):
        super(UNet_Discriminator, self).__init__()

        self.ch = D_ch
        self.resolution = resolution
        self.attention = D_attn
        self.n_classes = n_classes
        self.activation = D_activation
        self.init = D_init
        self.D_param = D_param

        self.lr = D_lr
        self.B1 = D_B1
        self.B2 = D_B2

        if self.resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]

        self.arch = D_unet_arch(self.ch, self.attention)[resolution]

        self.which_conv = functools.partial(SNConv2d, kernel_size=3, padding=1, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs)
        self.which_linear = functools.partial(SNLinear, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs)
        self.which_embedding = functools.partial(SNEmbedding, num_svs=num_D_SVs, num_itrs=num_D_SV_itrs)

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            if self.arch['downsample'][index]:
                self.blocks += [[
                    DBlock(in_channels=self.arch['in_channels'][index],
                           out_channels=self.arch['out_channels'][index],
                           which_conv=self.which_conv,
                           activation=self.activation,
                           preactivation=(index>0),
                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))
                ]]
            elif self.arch['upsample'][index]:
                upsample_function = functools.partial(F.interpolate, scale_factor=2, mode='nearest')
                self.blocks += [[
                    GBlock2(in_channels=self.arch['in_channels'][index],
                            out_channels=self.arch['out_channels'][index],
                            which_conv=self.which_conv,
                            activation=self.activation,
                            upsample=upsample_function)
                ]]

            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition:
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.blocks.append(
            nn.Conv2d(self.ch, 1, 1, 1, 0)
        )

        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        self.linear_middle = self.which_linear(16*self.ch, output_dim)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                init.orthogonal_(module.weight)

    def forward(self, x):
        h = x
        residual_features = [x]

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index == 6:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 7:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 8:
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 9:
                    h = torch.cat((h, residual_features[1]), dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index == self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)

        out = self.blocks[-1](h)
        out = out.view(out.size(0), 1, self.resolution, self.resolution)

        return out, bottleneck_out


if __name__ == "__main__":
    generator = Generator()
    discriminator = UNet_Discriminator()

    z = torch.randn(10, 128)
    out1 = generator(z)  # (10, 3, 128, 128)
    out2, bottleneck = discriminator(out1)

    print(out1.shape)
    print(out2.shape)
    print(bottleneck.shape)
