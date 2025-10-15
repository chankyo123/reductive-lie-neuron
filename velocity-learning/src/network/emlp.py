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
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        self.emlp = EMLP_Block(rep_in, rep_out, channels=64, G=G)
        # self.emlp = EMLP_Block(rep_in, rep_out, channels=128, G=G).to(device)
        # self.emlp = EMLP_Block(rep_in, rep_out, channels, G=SO(3)).to(device)
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

class EMLP_BasicBlock1D(nn.Module):
    """ EMLP를 사용한 1D ResNet의 기본 블록 (BatchNorm 제거) """
    expansion = 1

    def __init__(self, rep, stride=1, downsample=None):
        super(EMLP_BasicBlock1D, self).__init__()
        # G = rep.group
        G = SO(3)
        
        self.conv1 = EMLP_Conv1x1(rep, rep, G=G, stride=stride)
        # self.relu1 = EMLP_Block(rep, rep, channels=64, G=G)
        # self.conv2 = EMLP_Conv1x1(rep, rep, G=G, stride=1)
        self.relu2 = EMLP_Block(rep, rep, channels=64, G=G)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        # out = out.permute(0, 2, 1)
        # out = self.relu1(out)
        # out = out.permute(0, 2, 1)
        
        # out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out_permuted = out.permute(0, 2, 1)
        identity_permuted = identity.permute(0, 2, 1)
        
        out_permuted += identity_permuted
        out_permuted = self.relu2(out_permuted)
        
        return out_permuted.permute(0, 2, 1)
    
    
class SO3EquivariantResNetEMLP(nn.Module):
    def __init__(self, in_channels, out_dim=3, group_sizes=[2, 2, 2, 2]):
        super(SO3EquivariantResNetEMLP, self).__init__()
        
        self.G = SO(3)
        
        # 1. Representation을 추상적인 '설계도'로 먼저 정의합니다.
        rep_in_abstract = V
        rep_out_abstract = V
        
        # 2. 크기를 확인할 때는 그룹을 적용하여 구체화한 후 확인합니다.
        assert rep_in_abstract(self.G).size() == in_channels, \
            f"Input channels ({in_channels}) must match representation size ({rep_in_abstract(self.G).size()})"
        
        # 3. 클래스 변수에는 하위 모듈에 전달할 추상적인 Representation을 저장합니다.
        #    (EMLP_Block 같은 하위 모듈이 내부에서 그룹과 결합하기 때문)
        self.rep_in = rep_in_abstract
        self.rep_out = rep_out_abstract
        
        self.initial_conv = EMLP_Conv1x1(self.rep_in, self.rep_in, G=self.G, stride=2)
        # self.initial_relu = EMLP_Block(self.rep_in, self.rep_in, channels=64, G=self.G)
        self.pool = VNMeanPool_local(2)
        # self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.residual_groups1 = self._make_residual_group(self.rep_in, group_sizes[0], stride=1)
        self.residual_groups2 = self._make_residual_group(self.rep_in, group_sizes[1], stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.final_fc = EMLP_Block(self.rep_in, self.rep_out, channels=64, G=self.G)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _make_residual_group(self, rep, group_size, stride=1):
        downsample = None
        if stride != 1:
            downsample = EMLP_Conv1x1(rep, rep, G=self.G, stride=stride)

        layers = [EMLP_BasicBlock1D(rep, stride=stride, downsample=downsample)]
        for _ in range(1, group_size):
            layers.append(EMLP_BasicBlock1D(rep))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x_permuted = x.permute(0, 2, 1)
        # x_permuted = self.initial_relu(x_permuted)
        x = x_permuted.permute(0, 2, 1)
        x = self.pool(x)

        # device = torch.device("cpu")
        # x = self.residual_groups1(x.to(device))
        x = self.residual_groups1(x)
        # x = self.residual_groups2(x)

        x = self.global_avg_pool(x).squeeze(-1)
        mean = self.final_fc(x)
        # print(mean.shape)
        # assert False
        return mean