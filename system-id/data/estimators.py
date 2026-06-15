import torch


def estimate_A_ls(xs, ridge=1e-4):
    """Ridge-regularized least-squares estimate from one short trajectory."""
    X_prev = xs[:-1].T
    X_next = xs[1:].T
    n = X_prev.shape[0]
    G = X_prev @ X_prev.T + ridge * torch.eye(n, dtype=xs.dtype, device=xs.device)
    A_hat = X_next @ X_prev.T @ torch.linalg.inv(G)
    return A_hat
