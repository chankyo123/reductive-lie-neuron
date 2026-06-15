import math
import torch


def _randn(shape, generator=None, device='cpu', dtype=torch.float32):
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def sample_A(n=3, stable_prob=0.7, center_scale=0.35, generator=None):
    """Sample a 3x3 linear system with meaningful center (trace) and traceless parts."""
    M = _randn((n, n), generator=generator)
    tr = torch.trace(M) / float(n)
    A0 = M - tr * torch.eye(n)

    eigvals = torch.linalg.eigvals(A0).abs()
    radius = float(torch.max(eigvals.real).item())
    radius = max(radius, 1e-6)
    A0 = A0 / radius

    if torch.rand(1, generator=generator).item() < stable_prob:
        scale = torch.empty(1).uniform_(0.45, 0.95).item() if generator is None else (0.45 + (0.95 - 0.45) * torch.rand(1, generator=generator).item())
    else:
        scale = torch.empty(1).uniform_(1.02, 1.35).item() if generator is None else (1.02 + (1.35 - 1.02) * torch.rand(1, generator=generator).item())

    alpha = center_scale * (_randn((1,), generator=generator).item())
    A = scale * A0 + alpha * torch.eye(n)
    return A


def _random_orthogonal(n, generator=None):
    M = _randn((n, n), generator=generator)
    Q, R = torch.linalg.qr(M)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    Q = Q @ torch.diag(d)
    return Q


def sample_T(kind='gl', n=3, severity=1.0, generator=None):
    """
    Similarity transform sampler.
    kind='gl'      : orthogonal mixing + anisotropic scaling + shear
    kind='glhard'  : stronger version of 'gl'
    kind='scale'   : anisotropic scaling only (global isotropic scale removed)
    kind='shear'   : shear only
    kind='id'      : identity
    """
    if kind == 'id':
        return torch.eye(n)

    hard_mult = 2.0 if kind == 'glhard' else 1.0
    sev = severity * hard_mult

    Ql = _random_orthogonal(n, generator=generator)
    Qr = _random_orthogonal(n, generator=generator)

    # anisotropic scaling with determinant 1 (remove trivial global scale)
    s = _randn((n,), generator=generator) * (0.45 * sev)
    s = s - s.mean()  # sum zero -> determinant 1
    D = torch.diag(torch.exp(s))

    # upper triangular shear with ones on diagonal
    H = torch.eye(n)
    shear_mag = 0.35 * sev
    if kind in ['gl', 'glhard', 'shear']:
        for i in range(n):
            for j in range(i + 1, n):
                H[i, j] = shear_mag * _randn((1,), generator=generator).item()

    if kind == 'scale':
        T = Ql @ D @ Qr
    elif kind == 'shear':
        T = Ql @ H @ Qr
    else:
        T = Ql @ H @ D @ Qr
    return T


def conjugate(T, A):
    return T @ A @ torch.linalg.inv(T)
