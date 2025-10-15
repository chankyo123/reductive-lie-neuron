import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch

from core.lie_neurons_layers import *
from core.lie_alg_util import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance


import torch

def find_nearest_neighbors_batch(x, k=2):
    """
    Find nearest neighbors in point cloud using PyTorch.
    
    Args:
        x (torch.Tensor): Shape (1, 3, 100) point cloud.
        k (int): Number of neighbors.
    
    Returns:
        torch.Tensor: Shape (1, 2, 100) with nearest neighbor indices.
    """
    # Transpose to shape (1, 100, 3)
    x = x.permute(0, 2, 1)  # Shape: (1, 100, 3)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(x, x, p=2)  # Shape: (1, 100, 100)
    
    # Set diagonal to infinity to avoid self-neighbors
    batch_size, num_points, _ = dist_matrix.shape
    mask = torch.eye(num_points, device=x.device).bool().unsqueeze(0).expand(batch_size, -1, -1)
    dist_matrix[mask] = float('inf')
    
    # Get k nearest neighbors
    neighbors = torch.topk(dist_matrix, k, dim=-1, largest=False).indices  # (1, 100, k)
    
    # Transpose to shape (1, 2, 100)
    neighbors = neighbors.permute(0, 2, 1)
    
    return neighbors


def compute_vectors(points, neighbors):
    """
    Compute difference vectors from points to their neighbors using PyTorch.

    Args:
        points (torch.Tensor): Point cloud of shape (1, 3, 100).
        neighbors (torch.Tensor): Neighborhood indices of shape (1, 2, 100).

    Returns:
        torch.Tensor: Difference vectors of shape (1, 3, 2, 100).
    """
    # Transpose points to (1, 100, 3) for neighborhood indexing
    points_t = points.permute(0, 2, 1)  # (1, 100, 3)

    n, num_points, dim = points_t.shape  # n=1, num_points=100, dim=3
    k = neighbors.shape[1]  # k=2 neighbors

    # Initialize tensor for vectors
    vectors = torch.zeros((n, dim, k, num_points), device=points.device)

    # Compute vectors
    for i in range(n):
        for j in range(num_points):
            neighbor_indices = neighbors[i, :, j]  # Shape: (2,)
            
            # Reference point
            point = points_t[i, j].unsqueeze(0)  # Shape: (1, 3)
            
            # Neighbor points
            neighbor_points = points_t[i, neighbor_indices]  # Shape: (2, 3)

            # Compute difference vectors
            vecs = (neighbor_points - point).T  # Shape: (3, 2)
            
            # Store result
            vectors[i, :, :, j] = vecs

    return vectors

def compute_initial_vectors(vectors):
    for i, vec_set in enumerate(vectors):
        v1, v2 = vec_set[0], vec_set[1]
        mat_v1 = skew_symmetric(v1)
        mat_v2 = skew_symmetric(v2)
        mat_v3 = lie_bracket(mat_v1, mat_v2)
        # Return the initial 3 matrices
        return [mat_v1, mat_v2, mat_v3]
    return []



def lie_bracket(A, B):
    return torch.matmul(A, B) - torch.matmul(B, A)

def skew_symmetric(v):
    batch_size = v.shape[0]
    zero = torch.zeros(batch_size, device=v.device)

    # Construct skew-symmetric matrices
    skew = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
    ], dim=1)

    return skew

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
    num_features = 1
    out_features = 1

    x = torch.Tensor(np.random.rand(num_features, lie_dim, num_points)
                     )
    nn = find_nearest_neighbors_batch(x)
    vecs = compute_vectors(x, nn) #[1, 3, 2, 100]
    # Matrix Generators for gl(4)
    E11 = np.array([[1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E12 = np.array([[0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E13 = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E14 = np.array([[0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E21 = np.array([[0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E22 = np.array([[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E23 = np.array([[0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E24 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

    E31 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0]])

    E32 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0]])

    E33 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0]])

    E34 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]])

    E41 = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0]])

    generators = [E11, E12, E13, E14, E21, E22, E23, E24, E31, E32, E33, E34, E41]
    
    
    def compute_initial_vectors(vectors):
        """
        Compute initial skew-symmetric and Lie bracket matrices.

        Args:
            vectors (torch.Tensor): Shape (1, 3, 2, 100) with difference vectors.

        Returns:
            torch.Tensor: Shape (1, 3, 3, 3, 100) with the 3 matrices (v1, v2, v3).
        """
        batch_size, dim, k, num_points = vecs.shape  # (1, 3, 2, 100)
        
        assert k == 2, "Expected 2 neighbors for Lie bracket calculation."

        # Initialize output tensor for 3 matrices
        result = torch.zeros((batch_size, 3, 3, 3, num_points), device=vectors.device)

        for b in range(batch_size):
            for i in range(num_points):
                v1 = vectors[b, :, 0, i]  # (3,)
                v2 = vectors[b, :, 1, i]  # (3,)

                # Compute skew-symmetric matrices
                mat_v1 = skew_symmetric(v1.unsqueeze(0))[0]  # (3, 3)
                mat_v2 = skew_symmetric(v2.unsqueeze(0))[0]  # (3, 3)

                # Compute Lie bracket matrix
                mat_v3 = lie_bracket(mat_v1.unsqueeze(0), mat_v2.unsqueeze(0))[0]  # (3, 3)

                # Store the matrices in the output tensor
                result[b, 0, :, :, i] = mat_v1
                result[b, 1, :, :, i] = mat_v2
                result[b, 2, :, :, i] = mat_v3

        return result
    
    # Initial vectors
    initial_vectors = compute_initial_vectors(vecs)
    basis = initial_vectors.clone()
    print("b: ", vecs.shape, basis.shape)  #torch.Size([1, 3, 2, 100]) torch.Size([1, 3, 3, 3, 100])
    
    # Ensure exactly 16 vectors in the basis
    while len(basis) < 16:
        new_vectors = []
        for gen in generators:
            for b in basis.copy():  # Copy to avoid modifying the list during iteration
                if len(basis) >= 16:
                    break
                lie_bracket_res = lie_bracket(b, gen)
                if lie_bracket_res.any():  # Make sure we're not adding zero matrices
                    temp_basis = basis + [lie_bracket_res]
                    temp_matrix = np.array([v.flatten() for v in temp_basis]).T
                    if np.linalg.matrix_rank(temp_matrix) > len(basis):
                        basis.append(lie_bracket_res)
            if len(basis) >= 16:
                break
            

    x = x.reshape(1, num_features, lie_dim, num_points)
    y = torch.Tensor(np.random.rand(lie_dim))

    hat_layer = HatLayer(algebra_type=algebra_type)

    # Transformation
    Y = torch.linalg.matrix_exp(hat_layer(y))

    model = LNLinearAndKillingRelu(
        num_features, out_features, share_nonlinearity=True, algebra_type=algebra_type)
    # model = LN_SE(num_features, out_features, 4)
    # model = LN_ChannelSpatialAttention(num_features, 4)

    # print(x.shape) #[1, 1, 3, 100]
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
    test_equivariance('so3')  # Test for SL(3)
    # test_equivariance('sl3')  # Test for SL(3)
    # test_equivariance('sl4')  # Test for SL(4)
    # test_equivariance('gl3')  # Test for GL(3)
    # test_equivariance('gl4')  # Test for GL(4)