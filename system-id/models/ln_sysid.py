import torch
from models.sysid_common import BaseLieNet


class LNNet(BaseLieNet):
    def __init__(self, hid_c=16):
        super().__init__(hid_c=hid_c, algebra_type='sl3')

    def preprocess_input(self, x):
        trace = torch.diagonal(x, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        return x - (trace / 3.0) * eye
