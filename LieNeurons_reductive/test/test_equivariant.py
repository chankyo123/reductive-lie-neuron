import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch

from core.lie_neurons_layers import *
from core.lie_alg_util import *


def test_equivariance(algebra_type='sl3'):
    print(f"testing equivariant linear layer for {algebra_type.upper()}")
    
    # Set dimensions based on Lie type
    if algebra_type == 'gl4':
        lie_dim = 16
    elif algebra_type == 'gl3':
        lie_dim = 9
    elif algebra_type == 'sl4':
        lie_dim = 15
    elif algebra_type == 'sl3':
        lie_dim = 8
    elif algebra_type == 'so3':
        lie_dim = 3
    else:
        raise ValueError("Unsupported algebra type")
    
    num_points = 100
    num_features = 10
    out_features = 3

    x = torch.Tensor(np.random.rand(num_features, lie_dim, num_points)
                     ).reshape(1, num_features, lie_dim, num_points)
    y = torch.Tensor(np.random.rand(lie_dim))

    hat_layer = HatLayer(algebra_type=algebra_type)

    # Transformation
    Y = torch.linalg.matrix_exp(hat_layer(y))

    model = LNLinearAndKillingRelu(
        num_features, out_features, share_nonlinearity=True, algebra_type=algebra_type)
    # model = LN_SE(num_features, out_features, 4)
    # model = LN_ChannelSpatialAttention(num_features, 4)

    x_hat = hat_layer(x.transpose(2, -1))
    new_x_hat = torch.matmul(Y, torch.matmul(x_hat, torch.inverse(Y)))
    new_x = vee(new_x_hat, algebra_type=algebra_type).transpose(2, -1)

    model.eval()
    with torch.no_grad():
        out_x = model(x)
        print("out x : ", out_x.shape)
        out_new_x = model(new_x)

    out_x_hat = hat_layer(out_x.transpose(2, -1))
    out_x_hat_conj = torch.matmul(Y, torch.matmul(out_x_hat, torch.inverse(Y)))
    out_x_conj = vee(out_x_hat_conj, algebra_type=algebra_type).transpose(2, -1)

    # test_result = torch.allclose(
    #     out_new_x, out_x_conj, rtol=1e-4, atol=1e-4)
    rtol = 1e-4
    atol = 1e-4
    abs_diff = torch.abs(out_new_x - out_x_conj)
    tolerance = atol + rtol * torch.abs(out_x_conj)
    exceeds_tolerance = abs_diff > tolerance
    num_exceeding = exceeds_tolerance.sum().item()
    total_elements = out_new_x.numel()
    ratio_exceeding = num_exceeding / total_elements if total_elements > 0 else 0.0
    

    # Optional: Overall test result (True if no elements exceed tolerance)
    test_result = num_exceeding == 0


    print("out x[0,0,:,0]", out_x[0, 0, :, 0])
    print("out x conj[0,0,:,0]: ", out_x_conj[0, 0, :, 0])
    print("out new x[0,0,:,0]: ", out_new_x[0, 0, :, 0])
    print("differences: ",
          out_x_conj[0, 0, :, 0] - out_new_x[0, 0, :, 0])

    print("The network is equivariant: ", test_result)
    print(f"Number of elements exceeding tolerance: {num_exceeding} out of {total_elements}")
    print(f"Ratio exceeding tolerance: {ratio_exceeding:.6f}")


if __name__ == "__main__":
    # test_equivariance('so3')  # Test for SL(3)
    # test_equivariance('sl3')  # Test for SL(3)
    # test_equivariance('sl4')  # Test for SL(4)
    test_equivariance('gl3')  # Test for GL(3)
    # test_equivariance('gl4')  # Test for GL(4)