import sys
sys.path.append('.')

import numpy as np
import torch

from core.lie_neurons_layers import *
from core.lie_alg_util import *

def test_equivariance(algebra_type='gl2'):
    print(f"Testing equivariant linear layer for {algebra_type.upper()}")
    
    # Set dimensions based on Lie type
    if algebra_type == 'gl4':
        lie_dim = 16
    elif algebra_type == 'gl3':
        lie_dim = 9
    elif algebra_type == 'gl2':
        lie_dim = 4
    elif algebra_type == 'sl4':
        lie_dim = 15
    elif algebra_type == 'sl3':
        lie_dim = 8
    elif algebra_type == 'so3':
        lie_dim = 3
    else:
        raise ValueError("Unsupported algebra type")
    
    num_points = 3
    num_features = 1
    out_features = 3

    # Strain values from the dataset at t=0
    eps_1 = [9.999451359324651e-09,1.5237589706174638e-13,-6.068921244192933e-13,-3.7757101304947573e-13]
    eps_2 = [9.99990195477079e-09,-4.4127925257882346e-15,-9.020157622562472e-14,-1.282141135753545e-13]
    eps_3 = [1.000014403230656e-08,5.1318203502905685e-14,-2.1893000503094964e-14,1.2971729598887853e-13]
    # eps_1= np.eye(2).reshape(-1)
    # eps_2= np.eye(2).reshape(-1)* 0.1
    # eps_3= np.eye(2).reshape(-1)* 0.2
    
    sig_1 = [497.40835990055257, -0.0009420393521659, -0.0039104520796208, 102.50321237357244]
    sig_2 = [102.55032498036208, -0.0107521355455995, -0.0077854447347206, 497.4286151908094]
    sig_3 = [0.050271721492856, 384.4297313615423, 384.4409240534487, -0.0192708472802663]

    # Create input tensor [1, 1, 4, 3]
    x_values = torch.tensor([eps_1, eps_2, eps_3], dtype=torch.float32).T  # [4, 3]
    x = x_values.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 3]

    # Random element in the Lie algebra for transformation
    y = torch.Tensor(np.random.rand(lie_dim))
    hat_layer = HatLayer(algebra_type=algebra_type)

    # Transformation in GL(2)
    Y = torch.linalg.matrix_exp(hat_layer(y))
    # angle_degrees = 30
    # angle_radians = torch.tensor(angle_degrees * (torch.pi / 180))
    # Y = torch.tensor([
    #     [torch.cos(angle_radians), -torch.sin(angle_radians)],
    #     [torch.sin(angle_radians), torch.cos(angle_radians)]
    # ])
    
    # Initialize the model
    model = LNLinearAndKillingRelu(
        num_features, out_features, share_nonlinearity=True, algebra_type=algebra_type)

    # Compute transformed input
    x_hat = hat_layer(x.transpose(2, -1))  # [1, 1, num_points, 2, 2]
    new_x_hat = torch.matmul(Y, torch.matmul(x_hat, torch.inverse(Y)))
    new_x = vee(new_x_hat, algebra_type=algebra_type).transpose(2, -1)  # [1, 1, 4, 3]

    # Evaluate model
    model.eval()
    with torch.no_grad():
        out_x = model(x)  # [1, out_features, lie_dim, num_points]
        print("out_x shape: ", out_x.shape)
        out_new_x = model(new_x)

    # Compute conjugate of output
    out_x_hat = hat_layer(out_x.transpose(2, -1))
    out_x_hat_conj = torch.matmul(Y, torch.matmul(out_x_hat, torch.inverse(Y)))
    out_x_conj = vee(out_x_hat_conj, algebra_type=algebra_type).transpose(2, -1)

    # Check equivariance
    rtol = 1e-4
    atol = 1e-4
    abs_diff = torch.abs(out_new_x - out_x_conj)
    tolerance = atol + rtol * torch.abs(out_x_conj)
    exceeds_tolerance = abs_diff > tolerance
    num_exceeding = exceeds_tolerance.sum().item()
    total_elements = out_new_x.numel()
    ratio_exceeding = num_exceeding / total_elements if total_elements > 0 else 0.0
    test_result = num_exceeding == 0

    # Print results for the first point
    print("out_x[0,0,:,0]: ", out_x[0, 0, :, 0])
    print("out_x_conj[0,0,:,0]: ", out_x_conj[0, 0, :, 0])
    print("out_new_x[0,0,:,0]: ", out_new_x[0, 0, :, 0])
    print("Differences: ", out_x_conj[0, 0, :, 0] - out_new_x[0, 0, :, 0])
    print("The network is equivariant: ", test_result)
    print(f"Number of elements exceeding tolerance: {num_exceeding} out of {total_elements}")
    print(f"Ratio exceeding tolerance: {ratio_exceeding:.6f}")

if __name__ == "__main__":
    test_equivariance('gl2')