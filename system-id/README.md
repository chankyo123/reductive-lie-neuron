# GL(3) Multi-Trajectory System Identification (No Config Version)

This project builds a rebuttal-friendly benchmark for matrix-valued system identification:

- canonical train split: **ID only**
- test splits: **ID / GL / GL-hard / scale / shear**
- models: **Avg-LS / Non-equiv / LN / ReLN**
- train loss: **matrix MSE only**
- trace is evaluated only at test time

## Task
For each system, generate short trajectories from

x_{t+1} = A x_t + eps_t

Estimate local least-squares matrices A_tilde_i from each trajectory, then learn

{A_tilde_i}_{i=1}^N -> A_hat.

## Why GL(3), anisotropic scaling, and shear?
Under a state-space basis change x' = T x, the same linear dynamics is represented by

A' = T A T^{-1}.

Therefore isotropic scaling T = c I is trivial because

(cI) A (cI)^{-1} = A.

So meaningful GL(3) diversity comes from:

- anisotropic scaling (different scale per axis)
- shear
- orthogonal basis mixing

The sampler uses T = Q_l H D Q_r where
- D is determinant-1 anisotropic scaling
- H is upper-triangular shear
- Q_l, Q_r are random orthogonal matrices

## Files
- `data/`: synthetic benchmark generation
- `models/noneq_sysid.py`: MLP baseline
- `models/ln_sysid.py`: strict sl(3) model (trace removed)
- `models/reln_sysid.py`: strict gl(3) model (trace preserved)
- `models/sysid_common.py`: LieBatchNorm + SafeLNKillingRelu + BaseLieNet

## Commands

### analytic baseline
python train.py --model avg_ls --train_transform id --train_size 2000 --test_size 500 --cpu

### non-equiv baseline
python train.py --model noneq --train_transform id --train_size 2000 --test_size 500 --epochs 60 --cpu

### LN baseline
python train.py --model ln --train_transform id --train_size 2000 --test_size 500 --epochs 60 --cpu

### ReLN
python train.py --model reln --train_transform id --train_size 2000 --test_size 500 --epochs 60 --cpu

## Notes
1. If LN/ReLN still become unstable, reduce `--lr` further to `5e-5`.
2. This version intentionally removes the Lie bracket from the benchmark model for stability.
3. The normalization is **not** the previous Frobenius-norm batch norm. Instead it uses:
   - trace BN for the invariant center scalar
   - channel-wise scaling for the traceless matrix part

## Expected pattern
- Avg-LS should work immediately.
- Non-equiv should fit ID and degrade on GL/GL-hard.
- LN should do reasonably on traceless part, but fail on trace-sensitive metrics.
- ReLN should be the only model that can preserve the center structurally.
