from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from data.system_sampler import sample_A, sample_T, conjugate
from data.trajectory import rollout
from data.estimators import estimate_A_ls


@dataclass
class DatasetConfig:
    num_systems: int = 2000
    split: str = 'train'              # train / id_test / gl_test / glhard_test / scale_test / shear_test
    n: int = 3
    num_traj: int = 8
    traj_len: int = 8
    noise_std: float = 0.02
    ridge: float = 1e-4
    stable_prob: float = 0.7
    center_scale: float = 0.35
    seed: int = 0
    canonical_systems: torch.Tensor = None  # [S, n, n]
    exact_pair_eval: bool = False           # if True, build GL samples from transformed ID trajectories


def _sample_canonical_bank(cfg, g):
    if cfg.canonical_systems is not None:
        return cfg.canonical_systems
    bank = []
    for _ in range(cfg.num_systems):
        bank.append(sample_A(
            n=cfg.n,
            stable_prob=cfg.stable_prob,
            center_scale=cfg.center_scale,
            generator=g,
        ))
    return torch.stack(bank, dim=0)


class SyntheticGLSysIDDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.samples = []
        g = torch.Generator().manual_seed(cfg.seed)

        split_to_transform = {
            'train': 'id',
            'id_test': 'id',
            'gl_test': 'gl',
            'glhard_test': 'glhard',
            'scale_test': 'scale',
            'shear_test': 'shear',
        }
        transform_kind = split_to_transform[cfg.split]

        canonical_bank = _sample_canonical_bank(cfg, g)

        for system_idx in range(canonical_bank.shape[0]):
            A_canon = canonical_bank[system_idx].clone()
            T = sample_T(kind=transform_kind, n=cfg.n, generator=g)
            A_true = conjugate(T, A_canon)

            local_estimates = []
            for traj_idx in range(cfg.num_traj):
                x0_canon = torch.randn(cfg.n, generator=g)
                if cfg.exact_pair_eval and transform_kind != 'id':
                    xs_canon = rollout(A_canon, x0_canon, L=cfg.traj_len, noise_std=cfg.noise_std, generator=g)
                    xs = torch.einsum('ij,tj->ti', T, xs_canon)
                else:
                    x0 = T @ x0_canon
                    xs = rollout(A_true, x0, L=cfg.traj_len, noise_std=cfg.noise_std, generator=g)

                A_tilde = estimate_A_ls(xs, ridge=cfg.ridge)
                local_estimates.append(A_tilde)

            A_locals = torch.stack(local_estimates, dim=0)
            self.samples.append({
                'system_idx': int(system_idx),
                'A_locals': A_locals.float(),
                'A_true': A_true.float(),
                'A_canon': A_canon.float(),
                'T': T.float(),
                'split': cfg.split,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
