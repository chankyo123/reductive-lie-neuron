import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_alg_util import *
from core.reln_layers import *


class ReLNPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(ReLNPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = ReLNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = ReLNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_fc3 = ReLNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_pooling = ReLNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = ReLNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        # x = self.ln_fc2(x)  # [B, F, 8, N]
        # x = self.ln_fc3(x)  # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class ReLNReluPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(ReLNReluPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = ReLNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_pooling = ReLNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = ReLNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class ReLNBracketPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(ReLNBracketPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = ReLNLinearAndLieBracket(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        # self.ln_fc2 = ReLNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_pooling = ReLNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = ReLNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class ReLNReluBracketPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(ReLNReluBracketPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = ReLNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = ReLNLinearAndLieBracket(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_pooling = ReLNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = ReLNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_fc2(x)  # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class ReLNBracketNoResidualConnectPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(ReLNBracketNoResidualConnectPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = ReLNLinearAndLieBracketNoResidualConnect(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        # self.ln_fc2 = ReLNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_pooling = ReLNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = ReLNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class MLP(nn.Module):
    def __init__(self, in_channels):
        super(MLP, self).__init__()
        feat_dim = 256
        self.fc = nn.Linear(in_channels, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(feat_dim, 3)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, N]
        '''
        B, F, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x_out = torch.reshape(self.fc_final(x), (B, 3))     # [B, cls]

        return x_out
