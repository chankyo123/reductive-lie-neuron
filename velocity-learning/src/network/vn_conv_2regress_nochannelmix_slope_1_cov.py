"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch.nn as nn
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross, get_lie_algebra_feature
from models.lie_alg_util import *
from models.lie_neurons_layers import *
import time
import torch
from fvcore.nn import FlopCountAnalysis

def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 1D convolution with kernel size 3 """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """ 1D convolution with kernel size 1 """
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def vn_conv1x1(in_planes, out_planes, stride=1):
    """ 1D vn_convolution with kernel size 1 """
    if stride == 1:
        conv1x1 = nn.Linear(in_planes, out_planes, bias=False)
    elif stride == 2:
        conv1x1 = nn.Sequential(
            nn.Linear(in_planes, out_planes, bias=False),
            VNMeanPool(stride),
        )
    # return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return conv1x1


class VN_BasicBlock1D_cov(nn.Module):
    """ Supports: groups=1, dilation=1 """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(VN_BasicBlock1D_cov, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv3x1(in_planes, planes, stride)
        # if stride == 1:
        #     self.conv1 = nn.Linear(in_planes, planes, bias=False)
        # elif stride == 2:
        #     self.conv1 = nn.Linear(in_planes, planes, bias=False)
        #     # self.conv1_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        #     self.conv1_pool = local_mean_pool
        # else:
        #     assert False
            
        # self.bn1 = nn.BatchNorm1d(planes)
        self.bn1 = VNBatchNorm(planes, dim=3)
                                             
        # self.relu = nn.ReLU(inplace=True)
        self.relu = VNLeakyReLU(planes,negative_slope=0.0)
        
        # print("info of conv : ", planes, planes * self.expansion, stride)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        # self.conv2 = nn.Linear(planes, planes * self.expansion, bias=False)
        
        # self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.bn2 = VNBatchNorm(planes * self.expansion, dim=3)
        
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        # x = x.unsqueeze(1) #[1024, 64, 50]
        identity = x

        # if self.stride == 1:
        #     x = torch.permute(x,(0,2,1,3))
        #     x = x.reshape(-1, x.size(2), x.size(3))
        #     out = self.conv1(x)
            
        #     # out = self.conv1(torch.transpose(x,1,-1))
        #     # out = out.transpose(1,-1)
        # elif self.stride == 2:
        #     # out = self.conv1(x.transpose(1,-1))
        #     out = self.conv1(torch.transpose(x,1,-1))
        #     out = out.transpose(1,-1)
        #     out = self.conv1_pool(out)
            
        # else:
        #     assert False
        
        x = torch.permute(x,(0,2,1,3))
        x = x.reshape(-1, x.size(2), x.size(3))
        out = self.conv1(x)
        out = out.reshape(-1, 3, out.size(1), out.size(2))
        out = torch.permute(out,(0,2,1,3))

        out = self.bn1(out)
        out = self.relu(out)

        out = torch.permute(out,(0,2,1,3))
        out = out.reshape(-1, out.size(2), out.size(3))
        out = self.conv2(out)
        out = out.reshape(-1, 3, out.size(1), out.size(2))
        out = torch.permute(out,(0,2,1,3))
        
        # print('shape of input x, size check for downsample: ', x.shape)          #[1024, 10, 6, 50]
        # print('shape of x after conv2, size check for downsample: ', out.shape)  #[1024, 21, 6, 25]
        
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample[0](x)
            identity = identity.reshape(-1, 3, identity.size(1), identity.size(2))
            identity = torch.permute(identity,(0,2,1,3))

            identity = self.downsample[1](identity)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x1(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FcBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim):
        super(FcBlock, self).__init__()
        self.in_channel = in_channel//3
        self.out_channel = out_channel//3
        self.prep_channel = 128//3
        self.fc_dim = 512//3
        self.in_dim = in_dim

        # prep layer2
        self.prep1 = nn.Conv1d(
            self.in_channel, self.prep_channel, kernel_size=1, bias=False
        )
        self.bn1 = VNBatchNorm(self.prep_channel, dim=3)
        # fc layers
        self.fc1 = nn.Linear(self.prep_channel * self.in_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc3 = nn.Linear(self.fc_dim, self.out_channel)
        self.relu = VNLeakyReLU(self.fc_dim,negative_slope=0.0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.prep1.to('cuda')(x)
        # print('x shape after residual block:  ', x.shape)  #[1024, 510, 7] -> [1024, 170, 3, 7]
        x = torch.permute(x,(0,2,1,3)) 
        x = x.reshape(-1,x.size(2),x.size(3))
        # print('x shape before prep1 : ', x.shape)  #[1024, 170, 3, 7] -> [1024, 510, 7] -> [1024*3, 170, 7]
        x = self.prep1(x)
        
        # print('x shape after prep1 : ', x.shape)  #[1024, 170, 3, 7] -> [1024, 510, 7] -> [1024*3, 170, 7]
        x = x.reshape(-1, 3, x.size(1), x.size(2))
        x = torch.permute(x,(0,2,1,3))
        
        # x = self.bn1.to('cuda')(x)
        x = self.bn1(x)
        # print('x shape after bn1 : ', x.shape)  #[1024, 42, 3, 7]
        x = torch.permute(x,(0,2,1,3)) 
        x = x.reshape(x.size(0),x.size(1), -1)
        # x = self.fc1.to('cuda')(x)
        # print('self.fc1 weight : ', self.fc1.weight.shape) #[512, 896]
        x = self.fc1(x)
        # print('x shape after fc1 : ', x.shape)  #[1024, 512] -> [1024, 3, 170]
        x = torch.permute(x,(0,2,1)) 
        x = self.relu(x)
        
        # print('x shape after relu : ', x.shape)  #[1024, 170, 3]
        # x = self.dropout(x)
        
        # x = self.fc2.to('cuda')(x)
        x = torch.permute(x,(0,2,1)) 
        x = self.fc2(x)
        # print('x shape after fc2 : ', x.shape)  #[1024, 3, 170]
        x = torch.permute(x,(0,2,1)) 
        x = self.relu(x)
        
        # x = self.dropout(x)   
        
        # x = self.fc3.to('cuda')(x)
        x = torch.permute(x,(0,2,1)) 
        x = self.fc3(x)
        x = torch.permute(x,(0,2,1)) 
        
        # # >>> SO(3) Equivariance Check
        # print('value x after layers : ',x[:1,:1,:])    
        # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # x = torch.matmul(rotation_matrix, x.permute(2,0,1).reshape(3,-1)).reshape(3,x.size(0),x.size(1)).permute(1,2,0)
        # print('rotated value x after layers : ', x[:1,:1,:]) 
        # # <<< SO(3) Equivariance Check
        
        x = x.squeeze(dim=1)
        
        # # >>> SO(3) Equivariance Check
        # print('value x after layers : ',x[:1,:])    
        # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # x = torch.matmul(rotation_matrix, x.permute(1,0).reshape(3,-1)).reshape(3,x.size(0)).permute(1,0)
        # print('rotated value x after layers : ', x[:1,]) 
        # # <<< SO(3) Equivariance Check
        
        # print('x shape after fc3 : ', x.shape)  #[1024, 3, 1]
        return x

class InvariantMLP(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64, output_dim=1):
        """
        input_channels: number of channels in invariant_feat (3 here)
        hidden_dim: hidden layer size
        output_dim: number of scalar outputs to scale equivariant output (1 for safe scaling)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # optional: constrain scaling to [0,1]
        )

    def forward(self, invariant_feat):
        """
        invariant_feat: (B, 3, 200)
        Returns: (B, output_dim)
        """
        # Aggregate over window dimension (mean or max)
        x = invariant_feat.mean(dim=-1)  # (B, 3)
        # Pass through MLP
        scale = self.mlp(x)               # (B, output_dim)
        return scale
    
class LNLinear_VNBatch_KillingRelu(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='so3', share_nonlinearity=False, leaky_relu=False,negative_slope=0.2):
        super(LNLinear_VNBatch_KillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.linear = LNLinear(in_channels, out_channels)
        self.ln_bn = VNBatchNorm(out_channels, dim=4)
        self.leaky_relu = LNKillingRelu(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, negative_slope=negative_slope)

    def forward(self, x, M1=torch.eye(3), M2=torch.eye(3)):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x = self.linear(x)
        x = self.ln_bn(x)
        x_out = self.leaky_relu(x)
        return x
def safe_eigh(x, chunk_size=4096):
    # 1. Ensure symmetry
    x = 0.5 * (x + x.transpose(-1, -2))

    # 2. Try GPU in chunks
    eigvals_list, eigvecs_list = [], []
    try:
        for chunk in torch.split(x, chunk_size):
            vals, vecs = torch.linalg.eigh(chunk)
            eigvals_list.append(vals)
            eigvecs_list.append(vecs)
        return torch.cat(eigvals_list), torch.cat(eigvecs_list)

    except RuntimeError as e:
        print(f"[safe_eigh] GPU failed ({e}), falling back to CPU...")
        x_cpu = x.cpu()
        vals_list, vecs_list = [], []
        for chunk in torch.split(x_cpu, chunk_size):
            vals, vecs = torch.linalg.eigh(chunk)
            vals_list.append(vals)
            vecs_list.append(vecs)
        return torch.cat(vals_list), torch.cat(vecs_list)
    
class VN_ResNet1D_cov(nn.Module):
    def __init__(self, 
                block_type,
                in_dim, out_dim, 
                group_sizes,
                inter_dim, zero_init_residual=True):
        super(VN_ResNet1D_cov, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        div = 3
        self.base_plane = 64 //div
        self.inplanes = self.base_plane

        self.num_m_feature = 64
        self.first_out_channel = 64
        self.inplanes = self.first_out_channel//div
        # self.ln_bracket = LNLinearAndLieBracket(in_dim//3, self.num_m_feature//3, share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.input_block_conv  = nn.Conv1d(in_dim//div, 64//div, kernel_size=7, stride=2, padding=3, bias=False)
        self.ln_conv = nn.Conv1d(in_dim//div, self.first_out_channel//div, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_block_bn = VNBatchNorm(self.base_plane, dim=4)
        self.input_block_relu = VNLeakyReLU(self.base_plane,negative_slope=0.0)
        # self.local_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.local_pool = VNMeanPool_local(2)
        
        # self.map_m_to_m1 = LNLinear_VNBatch_VNRelu(self.first_out_channel//div,self.first_out_channel//div, negative_slope=0.0)
        # self.map_m_to_m2 = LNLinear_VNBatch_Liebracket(256//div,256//div)

        self.output_block1 = FcBlock(512//3*3, out_dim, 7)
        
        self.residual_groups1 = self._make_residual_group1d(block_type, 64//3, group_sizes[0], stride=1)
        # self.residual_groups2 = self._make_residual_group1d(block_type, 128//3, group_sizes[1], stride=2)
        # self.residual_groups3 = self._make_residual_group1d(block_type, 256//3, group_sizes[2], stride=2)
        self.residual_groups4 = self._make_residual_group1d(block_type, 512//3, group_sizes[3], stride=2)
        self.inv_mlp = InvariantMLP(input_channels=3, hidden_dim=64, output_dim=1)
        self._initialize(zero_init_residual)
        
    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        # print(group_sizes[0],group_sizes[1],group_sizes[2],group_sizes[3])
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                VNBatchNorm(planes * block.expansion, dim=4), 
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))
        # print(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    
    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, VN_BasicBlock1D_cov):
                    nn.init.constant_(m.bn2.bn.weight, 0)
                    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        
        v = x[:, :3, :]             # (B, 3, N)
        sigma_flat = x[:, 3:, :]    # (B, 9, N)
        B, _, N = v.shape

        # --- Reconstruct covariance matrix and ensure symmetry ---
        sigma = sigma_flat.view(B, 3, 3, N)  # (B, 3, 3, N)
        # sigma = 0.5 * (sigma + sigma.transpose(1, 2))  # symmetric

        # --- Flatten batch and point dims for vectorized eigendecomposition ---
        sigma_vec = sigma.permute(0, 3, 1, 2).reshape(B * N, 3, 3)  # (B*N, 3, 3)
        sigma_vec = 0.5 * (sigma_vec + sigma_vec.transpose(-1, -2))


        # --- Batched eigendecomposition with simple gauge fix ---
        eigvals, eigvecs = torch.linalg.eigh(sigma_vec)  # (B*N, 3), (B*N, 3, 3)
        # eigvals, eigvecs = safe_eigh(sigma_vec)
        # eigvals = eigvals.to(v.device)
        # eigvecs = eigvecs.to(v.device)
        
        # Simple gauge fix: ensure first element of each eigenvector positive
        signs = torch.sign(eigvecs[:, 0, :]).clamp(min=1e-6)  # (B*N, 3)
        eigvecs = eigvecs * signs.unsqueeze(1)               # apply sign fix

        # --- Restore batch and point dims ---
        eigvals = eigvals.reshape(B, N, 3).permute(0, 2, 1)    # (B, 3, N)
        eigvecs = eigvecs.reshape(B, N, 3, 3).permute(0, 2, 3, 1)  # (B, 3, 3, N)

        # --- Construct equivariant feature: velocity + eigenvectors ---
        equivariant_feat = torch.cat([v.unsqueeze(2), eigvecs], dim=2)  # (B, 3, 4, N)

        # --- Invariant feature: eigenvalues ---
        invariant_feat = eigvals  # (B, 3, N)
        x = torch.reshape(equivariant_feat,(-1,equivariant_feat.size(2),equivariant_feat.size(3)))
        x = self.input_block_conv(x)
        x = torch.reshape(x,(-1, 3, x.size(1), x.size(2)))
        x = torch.permute(x,(0,2,1,3))
        x = self.input_block_bn(x)
        x = self.input_block_relu(x)
        b,c,h,w = x.shape
        x = self.local_pool(x.reshape(-1, h,w))  
        x = torch.reshape(x,(b,c,x.shape[1],x.shape[2]))
        x = self.residual_groups1(x)
        b,c,h,w = x.shape
        x = self.local_pool(x.reshape(-1, h,w))  
        x = torch.reshape(x,(b,c,x.shape[1],x.shape[2]))
        x = self.residual_groups4(x)
        b,c,h,w = x.shape
        x = self.local_pool(x.reshape(-1, h,w))  
        x = torch.reshape(x,(b,c,x.shape[1],x.shape[2]))
        out_eq = self.output_block1(x)  # mean
        scale = self.inv_mlp(invariant_feat)       # (B, 1)
        mean = out_eq * scale 

        # # >>> SO(3) Equivariance Check : (3,1) vector and (3,3) covariance
         
        # print('value x after layers : ',mean[:1,:])    
        # print()
        # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # mean_rot = torch.matmul(rotation_matrix, mean.permute(1,0)).permute(1,0)
        # print('rotated value x after layers : ', mean_rot[:1,]) 
        
        # #1. using tlio's cov
        # # print('covariance after layers : ',covariance[:1,:])    
        # # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # # covariance_rot = torch.matmul(torch.matmul(rotation_matrix, covariance.permute(1,0)).permute(1,0), rotation_matrix.T)
        # # print('rotated value x after layers : ', covariance_rot[:1,:]) 
        
        # #2. using 3*3 cov
        # # print('covariance after layers : ',covariance[:1,:,:])    
        # # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # # covariance_rot = torch.matmul(torch.matmul(rotation_matrix, covariance.permute(1,2,0)).permute(2,0,1), rotation_matrix.T)
        # # print('rotated value x after layers : ', covariance_rot[:1,:,:]) 
        # # <<< SO(3) Equivariance Check
        return mean