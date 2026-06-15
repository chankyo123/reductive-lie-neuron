import torch
import torch.nn as nn

from models.lie_alg_util import HatLayer, vee, killingform
from models.lie_neurons_layers import LNLinear


class LieBatchNorm(nn.Module):
    """
    Adjoint-equivariant normalization for sl3/gl3 features.
    For gl3: normalize the invariant trace scalar, and learn a channel-wise
    scale for the traceless matrix part.
    For sl3: only the traceless scaling branch remains.

    Input shape: [B, C, K, N]
    Output shape: [B, C, K, N]
    """
    def __init__(self, num_features, algebra_type='gl3', eps=1e-5, momentum=0.1):
        super().__init__()
        self.algebra_type = algebra_type
        self.hat = HatLayer(algebra_type)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        if algebra_type == 'gl3':
            self.trace_bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        else:
            self.trace_bn = None

    def forward(self, x):
        # x: [B, C, K, N]
        x_perm = x.permute(0, 1, 3, 2)                 # [B, C, N, K]
        matrix = self.hat(x_perm)                      # [B, C, N, 3, 3]

        trace = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1)  # [B, C, N]
        eye = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 1, 1, 3, 3)
        center = (trace / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
        traceless = matrix - center

        traceless_scaled = traceless * self.gamma

        if self.trace_bn is None:
            out_matrix = traceless_scaled
            out_vec = vee(out_matrix, self.algebra_type)           # [B, C, N, K]
            return out_vec.permute(0, 1, 3, 2)                    # [B, C, K, N]

        trace_bn = self.trace_bn(trace.reshape(trace.shape[0], trace.shape[1], trace.shape[2]))
        center_bn = (trace_bn / 3.0).unsqueeze(-1).unsqueeze(-1) * eye
        out_matrix = traceless_scaled + center_bn
        out_vec = vee(out_matrix, self.algebra_type)
        return out_vec.permute(0, 1, 3, 2)


class SafeLNKillingRelu(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False,
                 leaky_relu=False, negative_slope=0.2, clamp_value=5.0):
        super().__init__()
        self.share_nonlinearity = share_nonlinearity
        if share_nonlinearity:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer = HatLayer(algebra_type)
        self.algebra_type = algebra_type
        self.leaky_relu = leaky_relu
        self.negative_slope = negative_slope
        self.clamp_value = clamp_value

    def forward(self, x):
        # x: [B, C, K, N]
        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        x_t = x.transpose(2, -1)   # [B, C, N, K]
        d_t = d.transpose(2, -1)

        x_hat = self.HatLayer(x_t)
        d_hat = self.HatLayer(d_t)
        kf_xd = killingform(x_hat, d_hat, self.algebra_type)
        kf_xd = torch.clamp(kf_xd, min=-self.clamp_value, max=self.clamp_value)

        if self.leaky_relu:
            mask = (kf_xd <= 0).float()
            out = self.negative_slope * x_t + (1.0 - self.negative_slope) * (
                mask * x_t + (1.0 - mask) * (x_t - (-kf_xd) * d_t)
            )
        else:
            out = torch.where(kf_xd <= 0, x_t, x_t - (-kf_xd) * d_t)
        return out.transpose(2, -1)


class BaseLieNet(nn.Module):
    """
    PointNet-style set encoder for local matrix estimates.
    Input:  [B, N_local, 3, 3]
    Output: [B, 3, 3]
    """
    def __init__(self, hid_c=16, algebra_type='gl3'):
        super().__init__()
        self.algebra_type = algebra_type
        self.hat_layer = HatLayer(algebra_type)

        self.layer1_lin = LNLinear(1, hid_c)
        self.layer1_bn = LieBatchNorm(hid_c, algebra_type=algebra_type)
        self.layer1_relu = SafeLNKillingRelu(hid_c, algebra_type=algebra_type, leaky_relu=True, negative_slope=0.1)

        self.layer2_lin = LNLinear(hid_c, hid_c)
        self.layer2_bn = LieBatchNorm(hid_c, algebra_type=algebra_type)
        self.layer2_relu = SafeLNKillingRelu(hid_c, algebra_type=algebra_type, leaky_relu=True, negative_slope=0.1)

        self.decoder_lin = LNLinear(hid_c, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def preprocess_input(self, x):
        return x

    def forward(self, x):
        # x: [B, N, 3, 3]
        x = self.preprocess_input(x)
        x_vec = vee(x, self.algebra_type)                 # [B, N, K]
        x_vec = x_vec.transpose(1, 2).unsqueeze(1)        # [B, 1, K, N]

        feat = self.layer1_lin(x_vec)
        feat = self.layer1_bn(feat)
        feat = self.layer1_relu(feat)

        feat = self.layer2_lin(feat)
        feat = self.layer2_bn(feat)
        feat = self.layer2_relu(feat)

        pooled = feat.mean(dim=-1, keepdim=True)          # [B, hid_c, K, 1]
        out_vec = self.decoder_lin(pooled)                # [B, 1, K, 1]
        out_vec = out_vec.squeeze(1).squeeze(-1)          # [B, K]
        out_mat = self.hat_layer(out_vec)                 # [B, 3, 3]
        return out_mat
