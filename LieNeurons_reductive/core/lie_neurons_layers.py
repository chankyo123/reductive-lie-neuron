import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_alg_util import *

sys.path.append('.')


EPS = 1e-6


class LNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LNLinear, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: input of shape [B, F, 8, N]
        '''
        x_out = self.fc(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class LNKillingRelu(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False, leaky_relu=False, negative_slope=0.2):
        super(LNKillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer = HatLayer(algebra_type)
        self.algebra_type = algebra_type
        self.leaky_relu = leaky_relu
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        # B, F, _, N = x.shape
        x_out = torch.zeros_like(x)

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        x = x.transpose(2, -1)
        d = d.transpose(2, -1)

        x_hat = self.HatLayer(x)
        d_hat = self.HatLayer(d)
        kf_xd = killingform(x_hat, d_hat, self.algebra_type)
        
        if self.leaky_relu:
            mask = (kf_xd <= 0).float()
            x_out = self.negative_slope * x + (1-self.negative_slope ) \
                *(mask*x + (1-mask)*(x-(-kf_xd)*d))
        else:
            x_out = torch.where(kf_xd <= 0, x, x - (-kf_xd) * d)
        x_out = x_out.transpose(2, -1)

        return x_out


class LNLieBracket(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLieBracket, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer =  HatLayer(algebra_type)


    def forward(self, x):
        '''
        x: point features of shape [B, F, K, N]
        '''
        # B, F, _, N = x.shape

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)         # [B, F, K, N]
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)       # [B, F, K, N]
        d = d.transpose(2, -1)

        d2 = d2.transpose(2,-1)

        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        return x_out
    

class LNLieBracketNoResidualConnect(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLieBracketNoResidualConnect, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

        # torch.nn.init.uniform_(self.learn_dir.weight, a=0.0, b=0.5)
        # torch.nn.init.uniform_(self.learn_dir2.weight, a=0.0, b=0.5)

        self.HatLayer = HatLayer()
        self.relu = LNKillingRelu(
            in_channels, share_nonlinearity=share_nonlinearity)


    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        # B, F, _, N = x.shape

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)

        d = d.transpose(2, -1)
        d2 = d2.transpose(2,-1)

        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        x_out = vee(lie_bracket,self.algebra_type).transpose(2, -1)
        
        # print("avg XY: ", torch.mean(torch.matmul(d2_hat, d_hat)))
        # print("avg YX: ", torch.mean(torch.matmul(d_hat,d2_hat)))
        # print("avg out: ", torch.mean(x_out))

        # print("median XY: ", torch.median(torch.matmul(d2_hat, d_hat)))
        # print("median YX: ", torch.median(torch.matmul(d_hat,d2_hat)))
        # print("median out: ", torch.median(x_out))

        # print("max XY: ", torch.max(torch.matmul(d2_hat, d_hat)))
        # print("max YX: ", torch.max(torch.matmul(d_hat,d2_hat)))
        # print("max out: ", torch.max(x_out))

        # print("min XY: ", torch.min(torch.abs(torch.matmul(d2_hat, d_hat))))
        # print("min YX: ", torch.min(torch.abs(torch.matmul(d_hat,d2_hat))))
        # print("min out: ", torch.min(torch.abs(x_out)))

        # print("avg X: ", torch.mean(x))
        # print("median X: ", torch.median(x))
        # print("max X: ", torch.max(x))
        # print("min X: ", torch.min(torch.abs(x)))
        # print("--------------------------------------------")

        return x_out


class LNEquivairanChannelMixing(nn.Module):
    def __init__(self, in_channel) -> None:
        super(LNEquivairanChannelMixing).__init__()

        self.ln_linear_relu = LNLinearAndKillingRelu(in_channel, 3, algebra_type='so3', share_nonlinearity=False),
        self.ln_pooling = LNMaxPool(3)

    def forward(self, x):
        x = self.ln_linear_relu(x)  # B, F, K (3 for so(3)), N
        x = self.ln_pooling(x).squeeze(-1)      # B, F, K (3 for so(3))
        out = torch.matmul(x.transpose(1, 2), x)
        return out
    
class LNLieBracketChannelMix(nn.Module):
    def __init__(self, in_channels, algebra_type='so3', share_nonlinearity=False, residual_connect=True):
        super(LNLieBracketChannelMix, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type
        self.residual_connect = residual_connect

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer =  HatLayer(algebra_type)


    def forward(self, x, M1=torch.eye(3), M2=torch.eye(3)):
        '''
        x: point features of shape [B, F, K, N]
        '''
        # B, F, _, N = x.shape
        M1 = M1.to(x.device)
        M2 = M2.to(x.device)

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)     # [B, F, K, N]
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)   # [B, F, K, N]

        d = torch.einsum('d k, b f k n -> b f n d',M1,d)             # [B, F, N, K]
        d2 = torch.einsum('d k, b f k n -> b f n d',M2,d2)           # [B, F, N, K]                      

        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        if self.residual_connect:
            x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        else:
            x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        return x_out


class LNLinearAndKillingRelu(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False, leaky_relu=False,negative_slope=0.2):
        super(LNLinearAndKillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, negative_slope=negative_slope)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)

        return x_out

class LNLinearAndLieBracket(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLinearAndLieBracket, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.liebracket = LNLieBracket(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # Bracket
        x_out = self.liebracket(x)

        return x_out
    

class LNLinearAndLieBracketChannelMix(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='so3', share_nonlinearity=False, residual_connect=True):
        super(LNLinearAndLieBracketChannelMix, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.linear = LNLinear(in_channels, out_channels)
        self.liebracket = LNLieBracketChannelMix(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, residual_connect=residual_connect)

    def forward(self, x, M1, M2):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # Bracket
        x_out = self.liebracket(x, M1, M2)

        return x_out

class LNLinearAndLieBracketNoResidualConnect(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLinearAndLieBracketNoResidualConnect, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.liebracket = LNLieBracketNoResidualConnect(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity)
        
        # torch.nn.init.uniform_(self.linear.fc.weight, a=0.0, b=0.5)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # Bracket
        x_out = self.liebracket(x)

        return x_out

class LNMaxPool(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3',abs_killing_form=False):
        super(LNMaxPool, self).__init__()
        self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.absolute = abs_killing_form
        self.algebra_type = algebra_type
        self.hat_layer = HatLayer(algebra_type=algebra_type)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        B, F, K, N = x.shape

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        x_hat = self.hat_layer(x.transpose(2, -1))
        d_hat = self.hat_layer(d.transpose(2, -1))
        killing_forms = killingform(x_hat, d_hat, self.algebra_type).squeeze(-1)


        if not self.absolute:
            idx = killing_forms.max(dim=-1, keepdim=False)[1]
        else:
            idx = killing_forms.abs().max(dim=-1, keepdim=False)[1]

        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing="ij")\
            + (idx.reshape(B, F, 1).repeat(1, 1, K),)
        x_max = x[index_tuple]
        return x_max


class LNLinearAndKillingReluAndPooling(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False, abs_killing_form=False,
                 use_batch_norm=False, dim=5):
        super(LNLinearAndKillingReluAndPooling, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity)
        self.max_pool = LNMaxPool(
            out_channels, algebra_type=algebra_type, abs_killing_form=abs_killing_form)
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm = LNBatchNorm(out_channels, dim=dim)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)

        if self.use_batch_norm:
            x = self.batch_norm(x)

        # LeakyReLU
        x_out = self.leaky_relu(x)

        x_out = self.max_pool(x_out)
        return x_out


class LNBatchNorm(nn.Module):
    def __init__(self, num_features, dim, algebra_type='sl3', affine=False, momentum=0.1):
        super(LNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features, affine=affine, momentum=momentum)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features, affine=affine, momentum=momentum)

        self.hat_layer = HatLayer(algebra_type=algebra_type)
        self.algebra_type = algebra_type

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, K, N_samples, ...]
        '''

        x_hat = self.hat_layer(x.transpose(2, -1))
        kf = killingform(x_hat, x_hat,algebra_type=self.algebra_type)
        # b, f, n, _, _ = x_hat.shape
        # kf = rearrange(torch.det(
        #         rearrange(x_hat, 'b f n m1 m2 -> (b f n) m1 m2')), '(b f n) -> b f n 1', b=b, f=f, n=n)
        kf = torch.where(kf <= 0, torch.clamp(
            kf, max=-EPS), torch.clamp(kf, min=EPS))

        # kf = torch.clamp(torch.abs(kf), min=EPS)
        kf = kf.squeeze(-1)

        # kf = compute_killing_form(x, x) + EPS
        
        kf_bn = self.bn(kf)
        # kf_bn = torch.clamp(torch.abs(self.bn(kf)), min=EPS)
        
        kf = kf.unsqueeze(2)
        kf_bn = kf_bn.unsqueeze(2)
        x = x / kf * kf_bn

        return x


class LNInvariant(nn.Module):
    def __init__(self, in_channel, algebra_type='sl3', dir_dim=8, method='learned_killing'):
        super(LNInvariant, self).__init__()

        self.hat_layer = HatLayer(algebra_type=algebra_type)
        self.learned_dir = LNLinearAndKillingRelu(
            in_channel, dir_dim, share_nonlinearity=True)
        self.method = method
        self.algebra_type = algebra_type

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, K, N_samples, ...]
        '''
        x_hat = self.hat_layer(x.transpose(2, -1))
        if self.method == 'learned_killing':
            d_hat = self.hat_layer(self.learned_dir(x).transpose(2, -1))
            x_out = killingform(x_hat, d_hat, algebra_type=self.algebra_type,feature_wise=True)
        elif self.method == 'self_killing':
            x_out = killingform(x_hat, x_hat,algebra_type=self.algebra_type)
        elif self.method == 'det':
            b, f, n, _, _ = x_hat.shape
            x_out = rearrange(torch.det(
                rearrange(x_hat, 'b f n m1 m2 -> (b f n) m1 m2')), '(b f n) -> b f n 1', b=b, f=f, n=n)
        elif self.method == 'trace':
            x_out = (x_hat.transpose(-1, -2) *
                     x_hat).sum(dim=(-1, -2))[..., None]

        return x_out


class LN_SE(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, algebra_type='so3'):
        """
        Equivariant Squeeze-and-Excitation block for features in a Lie algebra.
        
        Args:
            channels (int): Number of feature channels (F).
            reduction (int): Reduction ratio for the bottleneck (default: 16).
            algebra_type (str): Type of Lie algebra ('so3', 'sl3', etc.).
        """
        super(LN_SE, self).__init__()
        self.channels = in_channels
        self.reduction = reduction
        self.algebra_type = algebra_type
        
        # Excitation network: reduce F -> F/r -> F
        self.global_pool = LNMaxPool(in_channels)
        self.fc1 = LNLinear(in_channels, in_channels // reduction)
        self.fc2 = LNLinear(in_channels // reduction, in_channels)
        
        # Activations
        self.relu = LNKillingRelu(in_channels // reduction, algebra_type=algebra_type, share_nonlinearity=False, leaky_relu=False, negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the equivariant SE (Squeeze-and-Excitation) block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, F, K, N].
        
        Returns:
            torch.Tensor: Recalibrated tensor of shape [B, F, K, N].
        """
        B, F, K, N = x.size()
        
        # SE block for SO(3) case
        
        squeeze = self.global_pool(x)
        y = self.fc1(squeeze)  # [B, F/r, K]
        y = self.relu(y)
        y = self.fc2(y)        # [B, F, K]
        y_squeezed = y.mean(dim=2)  # [B, F]
        weights = self.sigmoid(y_squeezed)  # [B, F]
        weights = weights.unsqueeze(2)  # [B, F, 1]
        output = y * weights
        x_expanded_alt = y.unsqueeze(3)          # Shape: [B, F, 3, 1]
        x_out = torch.tile(x_expanded_alt, (1, 1, 1, 10))  # Shape: [B, F, 3, 10]
        
        return x_out
    
class LN_ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction_factor=16, spatial_kernel_size=7, algebra_type='so3'):
        super(LN_ChannelSpatialAttention, self).__init__()
        
        # Channel Attention (CA) Module
        self.avg_pool = LNMeanPool() 
        self.max_pool = LNMaxPool(in_channels)
        
        # Fully connected layers for CA (two 1D convolutions)
        self.fc1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu = LNKillingRelu(in_channels, algebra_type=algebra_type, share_nonlinearity=False, leaky_relu=False, negative_slope=0.2)
        self.fc2 = nn.Conv1d(in_channels, in_channels // reduction_factor, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Spatial Attention (SA) Module
        self.conv_sa = nn.Conv1d(2, 1, kernel_size=spatial_kernel_size, padding=(spatial_kernel_size-1)//2, bias=False)

    def forward(self, x):
        """
        Forward pass implementing CA and SA modules.
        Input: x (tensor) - shape (batch_size, in_channels, height, width)
        Output: refined feature map (tensor) - same shape as input
        """
        # Channel Attention (CA)
        max_pooled = self.max_pool(x).unsqueeze(-1)  # Shape: (batch_size, in_channels, 1, 1)
        max_pooled = torch.permute(max_pooled,(0,2,1,3))
        max_pooled = max_pooled.reshape(-1, max_pooled.size(2), max_pooled.size(3))
        avg_pooled = self.avg_pool(x).unsqueeze(-1)  # Shape: (batch_size, in_channels, 1, 1)
        avg_pooled = torch.permute(avg_pooled,(0,2,1,3))
        avg_pooled = avg_pooled.reshape(-1, avg_pooled.size(2), avg_pooled.size(3))
        
        # Pass through fully connected layers
        avg_out = self.fc1(avg_pooled)
        avg_out = avg_out.reshape(-1, 3, avg_out.size(1), avg_out.size(2))
        avg_out = torch.permute(avg_out,(0,2,1,3))
        avg_out = self.relu(avg_out)
        avg_out = torch.permute(avg_out,(0,2,1,3))
        avg_out = avg_out.reshape(-1, avg_out.size(2), avg_out.size(3))
        avg_out = self.fc2(avg_out)
        avg_out = avg_out.reshape(-1, 3, avg_out.size(1), avg_out.size(2))
        avg_out = torch.permute(avg_out,(0,2,1,3)).squeeze(-1)
        
        max_out = self.fc1(max_pooled)
        max_out = max_out.reshape(-1, 3, max_out.size(1), max_out.size(2))
        max_out = torch.permute(max_out,(0,2,1,3))
        max_out = self.relu(max_out)
        max_out = torch.permute(max_out,(0,2,1,3))
        max_out = max_out.reshape(-1, max_out.size(2), max_out.size(3))
        max_out = self.fc2(max_out)
        max_out = max_out.reshape(-1, 3, max_out.size(1), max_out.size(2))
        max_out = torch.permute(max_out,(0,2,1,3)).squeeze(-1)
        
        # Combine and apply sigmoid to get attention map (AMC)
        amc = self.sigmoid((avg_out + max_out).unsqueeze(3))  # Shape: (batch_size, in_channels//κ, 1, 1)
        # amc = F.interpolate(amc, size=(x.size(2), x.size(3)), mode='nearest')  # Upscale to match input spatial dims
        amc = torch.tile(amc, (1, 5, 1, 100)) 
        
        # Modulate input with CA attention map
        fca = amc * x  # Element-wise multiplication (broadcasting), FCA = AMC(F) ⊗ F
        print((avg_out + max_out).shape, x.shape)
        return x
        
        # Spatial Attention (SA)
        avg_spatial = torch.mean(fca, dim=1, keepdim=True)  # Avg pooling across channels
        max_spatial, _ = torch.max(fca, dim=1, keepdim=True)  # Max pooling across channels
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)  # Concatenate: (batch_size, 2, height, width)
        
        # Reshape for 1D convolution over spatial dimensions
        b, c, k, n = spatial_input.size()
        # spatial_input = spatial_input.view(b, c, k * n)  # Shape: (batch_size, 2, height*width)
        spatial_input = torch.permute(spatial_input,(0,2,1,3))
        spatial_input = spatial_input.reshape(-1, spatial_input.size(2), spatial_input.size(3))
        ams = self.conv_sa(spatial_input)  # Apply 1D conv, Shape: (batch_size, 1, height*width)
        # ams = ams.view(b, 1, k, n)  # Reshape back to (batch_size, 1, height, width)
        ams = ams.reshape(-1, 3, ams.size(1), ams.size(2))
        ams = torch.permute(ams,(0,2,1,3))
        ams = self.sigmoid(ams)  # Attention map (AMS)
        
        # Modulate FCA with SA attention map
        fsa = ams * fca  # Element-wise multiplication, FSA = AMS(FCA) ⊗ FCA
        
        return fsa
    
class LNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(LNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max
# class LNMeanPool(nn.Module):
#     def __init__(self, kernel=2):
#         super(LNMeanPool, self).__init__()
#         self.kernel = kernel    
#     def forward(self, x):
#         B,C,N,W = x.shape
#         if isinstance(W, torch.Tensor):
#             W = W.item()
#         reduced_W = (W + 1)//2 if self.kernel == 2 and W % 2 == 1 else W //self.kernel
#         # output = torch.zeros(B,C,N,reduced_W).to('cuda')
#         output = torch.zeros(B,C,N,reduced_W)

#         for i in range(reduced_W):
#             local_region = x[:, :, :, i * self.kernel : (i+1) * self.kernel]
#             local_average = local_region.mean(dim=3, keepdim=False)
#             output[:,:,:,i] = local_average
#         return output

class LNMeanPool(nn.Module):
    def __init__(self):
        super(LNMeanPool, self).__init__()
    def forward(self, x):
        return torch.mean(x, dim=3)  # Mean over N (dim=3), preserving [B, F, K]
