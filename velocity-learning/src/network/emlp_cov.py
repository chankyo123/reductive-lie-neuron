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
import emlp.nn.pytorch as emlpnn
from emlp.reps import V, T
from emlp.groups import SO


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

def EMLP_Block(rep_in, rep_out, channels, G=SO(3)):
    """ EMLP 라이브러리의 기본 블록을 사용합니다. """
    return emlpnn.EMLP(rep_in, rep_out, group=G, num_layers=3, ch=channels)

class EMLP_Conv1x1(nn.Module):
    """ 
    시퀀스의 각 요소에 독립적으로 EMLP를 적용하여 Conv1d처럼 동작하게 합니다.
    Stride는 AvgPool1d로 구현합니다.
    """
    def __init__(self, rep_in, rep_out, G=SO(3), stride=1):
        super().__init__()
        self.rep_in_dim = rep_in(G).size()
        self.rep_out_dim = rep_out(G).size()
        self.emlp = EMLP_Block(rep_in, rep_out, channels=128, G=G)
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        B, C, N = x.shape
        assert C == self.rep_in_dim, f"Input channel {C} does not match rep dimension {self.rep_in_dim}"
        
        x = x.permute(0, 2, 1) # (B, C, N) -> (B, N, C)
        x = self.emlp(x)       # Output: (B, N, C_out)
        x = x.permute(0, 2, 1) # (B, N, C_out) -> (B, C_out, N)
        
        if self.stride > 1:
            x = self.pool(x)
        return x

class EMLP_BasicBlock1D_cov(nn.Module):
    """ EMLP를 사용한 1D ResNet의 기본 블록 (BatchNorm 제거) """
    expansion = 1

    def __init__(self, rep, stride=1, downsample=None):
        super(EMLP_BasicBlock1D_cov, self).__init__()
        G = rep.group
        
        self.conv1 = EMLP_Conv1x1(rep, rep, G=G, stride=stride)
        self.relu1 = EMLP_Block(rep, rep, channels=128, G=G)
        self.conv2 = EMLP_Conv1x1(rep, rep, G=G, stride=1)
        self.relu2 = EMLP_Block(rep, rep, channels=128, G=G)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = out.permute(0, 2, 1); out = self.relu1(out); out = out.permute(0, 2, 1)
        
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out_permuted = out.permute(0, 2, 1)
        identity_permuted = identity.permute(0, 2, 1)
        
        out_permuted += identity_permuted
        out_permuted = self.relu2(out_permuted)
        
        return out_permuted.permute(0, 2, 1)
    
    
class SO3EquivariantResNetEMLP_cov(nn.Module):
    def __init__(self, in_channels, out_dim=3, group_sizes=[2, 2, 2, 2]):
        super(SO3EquivariantResNetEMLP_cov, self).__init__()
        
        self.G = SO(3)
        
        # 1. Representation을 추상적인 '설계도'로 먼저 정의합니다.
        rep_in_abstract = V + T(2)
        rep_out_abstract = V
        
        # 2. 크기를 확인할 때는 그룹을 적용하여 구체화한 후 확인합니다.
        assert rep_in_abstract(self.G).size() == in_channels, \
            f"Input channels ({in_channels}) must match representation size ({rep_in_abstract(self.G).size()})"
        
        # 3. 클래스 변수에는 하위 모듈에 전달할 추상적인 Representation을 저장합니다.
        #    (EMLP_Block 같은 하위 모듈이 내부에서 그룹과 결합하기 때문)
        self.rep_in = rep_in_abstract
        self.rep_out = rep_out_abstract
        
        self.initial_conv = EMLP_Conv1x1(self.rep_in, self.rep_in, G=self.G, stride=2)
        self.initial_relu = EMLP_Block(self.rep_in, self.rep_in, channels=128, G=self.G)
        self.pool = VNMeanPool_local(2)
        # self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.residual_groups1 = self._make_residual_group(self.rep_in, group_sizes[0], stride=1)
        self.residual_groups2 = self._make_residual_group(self.rep_in, group_sizes[1], stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.final_fc = EMLP_Block(self.rep_in, self.rep_out, channels=256, G=self.G)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _make_residual_group(self, rep, group_size, stride=1):
        downsample = None
        if stride != 1:
            downsample = EMLP_Conv1x1(rep, rep, G=self.G, stride=stride)

        layers = [EMLP_BasicBlock1D_cov(rep, stride=stride, downsample=downsample)]
        for _ in range(1, group_size):
            layers.append(EMLP_BasicBlock1D_cov(rep))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x_permuted = x.permute(0, 2, 1)
        x_permuted = self.initial_relu(x_permuted)
        x = x_permuted.permute(0, 2, 1)
        x = self.pool(x)

        x = self.residual_groups1(x)
        x = self.residual_groups2(x)

        x = self.global_avg_pool(x).squeeze(-1)
        mean = self.final_fc(x)
        return mean

class EMLP_cov(nn.Module):
    def __init__(self, 
                block_type,
                in_dim, out_dim, 
                group_sizes,
                inter_dim, zero_init_residual=True):
        super(EMLP_cov, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        div = 3
        
        self.num_m_feature = 64
        self.first_out_channel = 64
        self.inplanes = self.first_out_channel//div
        # self.ln_bracket = LNLinearAndLieBracket(in_dim//3, self.num_m_feature//3, share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.ln_linear = LNLinear(in_dim//3, self.first_out_channel//3)
        self.ln_conv = nn.Conv1d(in_dim//div, self.first_out_channel//div, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_block_bn = LNBatchNorm(self.first_out_channel//3)
        # self.input_block_relu = VNLeakyReLU(self.first_out_channel//3,negative_slope=0.0)
        self.liebracket = LNLieBracket(self.first_out_channel//3, algebra_type='gl3', share_nonlinearity=share_nonlinearity)
        
        self.ln_pool = VNMeanPool_local(2)
        
        # self.map_m_to_m1 = LNLinear_VNBatch_VNRelu(self.first_out_channel//div,self.first_out_channel//div, negative_slope=0.0)
        # self.map_m_to_m2 = LNLinear_VNBatch_Liebracket(256//div,256//div)

        self.output_block1 = FcBlock(512//3*3, out_dim, 7)
        
        self.residual_groups1 = self._make_residual_group1d(block_type, self.first_out_channel//3, group_sizes[0], stride=1)
        # self.residual_groups2 = self._make_residual_group1d(block_type, 128//3, group_sizes[1], stride=2)
        # self.residual_groups3 = self._make_residual_group1d(block_type, 256//3, group_sizes[2], stride=2)
        self.residual_groups4 = self._make_residual_group1d(block_type, 512//3, group_sizes[3], stride=2)
        # self.ln_conv2 = nn.Conv1d(512//3, 512//3, kernel_size=7, stride=2, padding=3, bias=False)
        
        # self.ln_fc = LNLinearAndKillingRelu(
        #     feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        # self.ln_fc2 = LNLinearAndKillingRelu(
        #     feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        # self.ln_fc_bracket2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        # self.fc_final = nn.Linear(feat_dim, 1, bias=False)
        self._initialize(zero_init_residual)
        
    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        # print(group_sizes[0],group_sizes[1],group_sizes[2],group_sizes[3])
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                LNBatchNorm(planes * block.expansion), 
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
            # print(type(m))
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, VNBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
        final_layer = list(self.modules())[-1] 
        if isinstance(final_layer, torch.nn.Linear):
            torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(final_layer.bias, 0.0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck1D):
                #     nn.init.constant_(m.bn3.weight, 0)
                # print(m)
                # if isinstance(m, LNLinearAndLieBracketChannelMix_VNBatchNorm):
                #     nn.init.constant_(m.ln_bn.bn.weight, 0)
                if isinstance(m, LN_BasicBlock1D_cov):
                    nn.init.constant_(m.bn2.bn.weight, 0)
                    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        
        # x = self.input_block_conv(x)
        x = x.unsqueeze(1)
        x = get_lie_algebra_feature(x)
        # x = x.permute(0, 3, 1, 2)   # when one single 3d input (w/ acc)
        
        x = torch.permute(x,(0,3,2,1))
        
        # x = torch.reshape(x,(-1,x.size(2),x.size(3)))
        # print(x.shape) #[1024, 3, 2, 200]
        
        
        # print(x.shape)
        # x = rearrange(x,'b c d f -> b d c f')
        # x = self.ln_bracket(x)
        # print(self.ln_bracket)
        # x = self.ln_pool(x)
        # x = self.ln_pool(x)
        
        # linear_flops = FlopCountAnalysis(self.ln_linear, x_linear)
        # print(f"linear-FLOPs: {linear_flops.total()}")   # 25804800

        # x = self.ln_linear(x)
        # x = self.ln_pool(x)
        # x = self.vn_bn(x)
        # print(x.shape)   #[1024, 3, 2, 200]
        # conv_flops = FlopCountAnalysis(self.ln_conv, x)
        # print(f"conv-FLOPs: {conv_flops.total()}")   #90316800
        
        x = torch.reshape(x,(-1,x.size(2),x.size(3)))
        x = self.ln_conv(x)
        x = torch.reshape(x,(-1, 9, x.size(1), x.size(2)))
        x = torch.permute(x,(0,2,1,3))
        x = self.input_block_bn(x)
        b,c,h,w = x.shape
        #1.
        # x = self.input_block_relu(x)
        #1.
        #2.
        x = self.liebracket(x)
        #2.
        x = self.ln_pool(x)
        
        # print(x.shape)  #[1024, 21, 3, 50] -> [1024, 85, 3, 50]
        x = torch.reshape(x,(b,c,h,-1))
        x = self.residual_groups1(x)
        # m1 = self.map_m_to_m1(x)
        # M1 = torch.einsum("b f k d, b k e d -> b f e d", m1, m1.transpose(1,2))
        # x = torch.einsum("b f k d, b k e d -> b f e d", M1,x)
        
        # x = self.residual_groups2(x)
        # x = self.residual_groups3(x)
        
        # m2 = self.map_m_to_m2(x)
        # M2 = torch.einsum("b f k d, b k e d -> b f e d", m2, m2.transpose(1,2))
        # x = torch.einsum("b f k d, b k e d -> b f e d", M2,x)
        x = self.ln_pool(x)
        
        x = self.residual_groups4(x)
        
        x = self.ln_pool(x)


        # x = rearrange(x,'b f w d -> (b w) f d')
        # x = self.ln_conv2(x)
        # x = torch.reshape(x,(-1, 3, x.size(1), x.size(2)))
        # x = torch.permute(x,(0,2,1,3))
        
        # x = self.ln_channel_mix_residual2(x, M3, M4)
        # x = self.ln_channel_mix_residual3(x, M5, M6)
        # x = self.ln_channel_mix_residual4(x, M7, M8)
        # x = rearrange(x,'b (f w) d 1 -> b f d w', w = 7)
        # print(x.shape)  #[1024, 170, 3, 7]

        mean = self.output_block1(x)  
        
        ## make left so(3) equivariant vector
        matrix_a = mean.view(-1, 3, 3)
        matrix_a_skew = 0.5 * (matrix_a - matrix_a.transpose(-2, -1))
        # x = matrix_a_skew[:, 1, 0]
        # y = matrix_a_skew[:, 0, 2]
        # z = matrix_a_skew[:, 2, 1]
        x = matrix_a_skew[:, 2, 1]
        y = matrix_a_skew[:, 0, 2]
        z = matrix_a_skew[:, 1, 0]
        mean = torch.stack((x, y, z), dim=1)
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