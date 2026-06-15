import torch


def mat_trace(A):
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)


def traceless_part(A):
    n = A.shape[-1]
    tr = mat_trace(A)
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    return A - (tr[..., None, None] / float(n)) * I


def conjugate(T, A):
    return T @ A @ torch.linalg.inv(T)


def pullback_to_canonical(A_pred, T):
    Tinv = torch.linalg.inv(T)
    return Tinv @ A_pred @ T


def matrix_mse(A_pred, A_true):
    return ((A_pred - A_true) ** 2).sum(dim=(-2, -1)).mean()


def trace_mse(A_pred, A_true):
    return ((mat_trace(A_pred) - mat_trace(A_true)) ** 2).mean()


def traceless_mse(A_pred, A_true):
    return ((traceless_part(A_pred) - traceless_part(A_true)) ** 2).sum(dim=(-2, -1)).mean()


def relative_fro_error(A_pred, A_true):
    num = torch.norm(A_pred - A_true, dim=(-2, -1))
    den = torch.norm(A_true, dim=(-2, -1)).clamp_min(1e-8)
    return (num / den).mean()


def canonical_relative_fro_error(A_pred, A_canon, T):
    A_pb = pullback_to_canonical(A_pred, T)
    return relative_fro_error(A_pb, A_canon)


def rollout_error(A_pred, A_true, steps=10):
    B, n, _ = A_pred.shape
    x0 = torch.randn(B, n, device=A_pred.device, dtype=A_pred.dtype)
    xp = x0.clone()
    xt = x0.clone()
    err = 0.0
    for _ in range(steps):
        xp = (A_pred @ xp.unsqueeze(-1)).squeeze(-1)
        xt = (A_true @ xt.unsqueeze(-1)).squeeze(-1)
        err += ((xp - xt) ** 2).sum(dim=-1).mean()
    return err / float(steps)


def canonical_rollout_error(A_pred, A_canon, T, steps=10):
    A_pb = pullback_to_canonical(A_pred, T)
    return rollout_error(A_pb, A_canon, steps=steps)


def canonical_metrics(A_pred, A_canon, T):
    A_pb = pullback_to_canonical(A_pred, T)
    return {
        'canon_matrix_mse': float(matrix_mse(A_pb, A_canon).item()),
        'canon_trace_mse': float(trace_mse(A_pb, A_canon).item()),
        'canon_traceless_mse': float(traceless_mse(A_pb, A_canon).item()),
        'canon_rel_fro': float(relative_fro_error(A_pb, A_canon).item()),
        'canon_rollout_err': float(rollout_error(A_pb, A_canon).item()),
    }


def equivariance_consistency(A_pred, A_canon, T):
    A_push = conjugate(T, A_canon)
    return float(torch.norm(A_pred - A_push, dim=(-2, -1)).mean().item())


def summarize_predictions(A_pred, A_true, A_canon=None, T=None):
    out = {
        'matrix_mse': float(matrix_mse(A_pred, A_true).item()),
        'trace_mse': float(trace_mse(A_pred, A_true).item()),
        'traceless_mse': float(traceless_mse(A_pred, A_true).item()),
        'rel_fro': float(relative_fro_error(A_pred, A_true).item()),
        'rollout_err': float(rollout_error(A_pred, A_true).item()),
    }
    if A_canon is not None and T is not None:
        out.update(canonical_metrics(A_pred, A_canon, T))
        out['eq_consistency'] = equivariance_consistency(A_pred, A_canon, T)
    return out
