import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


########################################## Power Iteration Method functions ##########################################
# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x

# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
    # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


########################################## Spectral Normalization ##########################################
# Spectral Normalization
class SN:
    def __init__(self, num_svs, num_itrs, num_outputs):
        # Power Iteration Method: approximation of Singular value decomposition (SVD)
        # Number of singular values
        self.num_svs = num_svs

        # Number of power iterations per step
        self.num_itrs = num_itrs

        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)

        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training)

        # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, num_svs=1, num_itrs=1):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        SN.__init__(self, num_svs, num_itrs, out_channels)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride, self.padding)

    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_channels, out_channels, bias=True, num_svs=1, num_itrs=1):
        nn.Linear.__init__(self, in_channels, out_channels, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class SNEmbedding(nn.Embedding, SN):
    def __init__(self, num_embddings, embedding_dim, num_svs=1, num_itrs=1):
        nn.Embedding.__init__(num_embddings, embedding_dim)
        SN.__init__(num_svs, num_itrs, num_embddings)
    def forward(self, x):
        return F.embedding(x, self.W_())


########################################## Attention Blocks ##########################################
class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


########################################## Batch Norm Blocks ##########################################
class bn(nn.Module):
    def __init__(self, output_size):
        super(bn, self).__init__()

        self.output_size = output_size
        self.gain = Parameter(torch.ones(output_size), requires_grad=True)
        self.bias = Parameter(torch.ones(output_size), requires_grad=True)

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var',  torch.ones(output_size))

    def forward(self, x):
        return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain, self.bias)


########################################## Generator Blocks ##########################################
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=bn, activation=None, upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample

        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(self.in_channels, self.out_channels, kernel_size=1, padding=0)

        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        self.upsample = upsample

    def forward(self, x):  # x = (10, 1024, 4, 4)
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x  # x = (10, 1024, 8, 8)


########################################## UNet Discriminator Blocks ##########################################
class GBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=bn, activation=None, upsample=None):
        super(GBlock2, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample

        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(self.in_channels, self.out_channels, kernel_size=1, padding=0)

        self.upsample = upsample

    def forward(self, x):  # x =
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x  # x =


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, preactivation=False, activation=None, downsample=None):
        super(DBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation

        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or downsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(self.in_channels, self.out_channels, kernel_size=1, padding=0)

        self.downsample = downsample

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(x)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)

