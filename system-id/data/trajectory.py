import torch


def _randn_like(x, generator=None):
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def rollout(A, x0, L=8, noise_std=0.02, generator=None):
    xs = [x0]
    x = x0
    for _ in range(L):
        noise = noise_std * _randn_like(x, generator=generator)
        x = A @ x + noise
        xs.append(x)
    return torch.stack(xs, dim=0)
