import argparse
import copy
import json
import os

import torch
from torch.utils.data import DataLoader

from data.dataset import DatasetConfig, SyntheticGLSysIDDataset
from data.system_sampler import sample_A
from eval import eval_model
from metrics import matrix_mse
from models.noneq_sysid import NonEqNet
from models.ln_sysid import LNNet
from models.reln_sysid import ReLNNet
from utils import set_seed, ensure_dir


def build_model(name, hidden_dim=16):
    if name == 'noneq':
        return NonEqNet(hidden_dim=hidden_dim)
    if name == 'ln':
        return LNNet(hid_c=hidden_dim)
    if name == 'reln':
        return ReLNNet(hid_c=hidden_dim)
    raise ValueError(name)


def make_canonical_bank(num_systems, n, stable_prob, center_scale, seed):
    g = torch.Generator().manual_seed(seed)
    bank = []
    for _ in range(num_systems):
        bank.append(sample_A(n=n, stable_prob=stable_prob, center_scale=center_scale, generator=g))
    return torch.stack(bank, dim=0)


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        A_locals = batch['A_locals'].to(device)
        A_true = batch['A_true'].to(device)

        A_pred = model(A_locals)
        loss = matrix_mse(A_pred, A_true)

        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss encountered. Reduce lr or inspect model activations.')

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = A_locals.shape[0]
        total += loss.item() * bs
        count += bs
    return total / float(count)


def metric_subset(d):
    keys = ['matrix_mse','trace_mse','traceless_mse','rel_fro','rollout_err',
            'canon_matrix_mse','canon_trace_mse','canon_traceless_mse','canon_rel_fro','canon_rollout_err','eq_consistency']
    return {k: d[k] for k in keys if k in d}


def main(args):
    set_seed(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    ensure_dir(args.output_dir)

    train_bank = None
    if args.train_transform != 'id':
        train_bank = make_canonical_bank(args.train_size, 3, args.stable_prob, args.center_scale, args.seed)

    test_bank = make_canonical_bank(args.test_size, 3, args.stable_prob, args.center_scale, args.seed + 999)

    train_cfg = DatasetConfig(
        num_systems=args.train_size,
        split='train' if args.train_transform == 'id' else 'gl_test',
        n=3,
        num_traj=args.num_traj,
        traj_len=args.traj_len,
        noise_std=args.noise_std,
        ridge=args.ridge,
        stable_prob=args.stable_prob,
        center_scale=args.center_scale,
        seed=args.seed,
        canonical_systems=train_bank,
        exact_pair_eval=False,
    )
    id_cfg = DatasetConfig(num_systems=args.test_size, split='id_test', n=3, num_traj=args.num_traj, traj_len=args.traj_len, noise_std=args.noise_std, ridge=args.ridge, stable_prob=args.stable_prob, center_scale=args.center_scale, seed=args.seed + 101, canonical_systems=test_bank, exact_pair_eval=args.exact_pair_eval)
    gl_cfg = DatasetConfig(num_systems=args.test_size, split='gl_test', n=3, num_traj=args.num_traj, traj_len=args.traj_len, noise_std=args.noise_std, ridge=(0.0 if args.exact_pair_eval else args.ridge), stable_prob=args.stable_prob, center_scale=args.center_scale, seed=args.seed + 202, canonical_systems=test_bank, exact_pair_eval=args.exact_pair_eval)
    gh_cfg = DatasetConfig(num_systems=args.test_size, split='glhard_test', n=3, num_traj=args.num_traj, traj_len=args.traj_len, noise_std=args.noise_std, ridge=(0.0 if args.exact_pair_eval else args.ridge), stable_prob=args.stable_prob, center_scale=args.center_scale, seed=args.seed + 303, canonical_systems=test_bank, exact_pair_eval=args.exact_pair_eval)
    sc_cfg = DatasetConfig(num_systems=args.test_size, split='scale_test', n=3, num_traj=args.num_traj, traj_len=args.traj_len, noise_std=args.noise_std, ridge=(0.0 if args.exact_pair_eval else args.ridge), stable_prob=args.stable_prob, center_scale=args.center_scale, seed=args.seed + 404, canonical_systems=test_bank, exact_pair_eval=args.exact_pair_eval)
    sh_cfg = DatasetConfig(num_systems=args.test_size, split='shear_test', n=3, num_traj=args.num_traj, traj_len=args.traj_len, noise_std=args.noise_std, ridge=(0.0 if args.exact_pair_eval else args.ridge), stable_prob=args.stable_prob, center_scale=args.center_scale, seed=args.seed + 505, canonical_systems=test_bank, exact_pair_eval=args.exact_pair_eval)

    train_set = SyntheticGLSysIDDataset(train_cfg)
    id_set = SyntheticGLSysIDDataset(id_cfg)
    gl_set = SyntheticGLSysIDDataset(gl_cfg)
    gh_set = SyntheticGLSysIDDataset(gh_cfg)
    sc_set = SyntheticGLSysIDDataset(sc_cfg)
    sh_set = SyntheticGLSysIDDataset(sh_cfg)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    id_loader = DataLoader(id_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gl_loader = DataLoader(gl_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gh_loader = DataLoader(gh_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sc_loader = DataLoader(sc_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    sh_loader = DataLoader(sh_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == 'avg_ls':
        results = {
            'id': metric_subset(eval_model(None, id_loader, device, analytic=True)),
            'gl': metric_subset(eval_model(None, gl_loader, device, analytic=True)),
            'glhard': metric_subset(eval_model(None, gh_loader, device, analytic=True)),
            'scale': metric_subset(eval_model(None, sc_loader, device, analytic=True)),
            'shear': metric_subset(eval_model(None, sh_loader, device, analytic=True)),
        }
        print(json.dumps(results, indent=2))
        return

    model = build_model(args.model, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_gl = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        id_metrics = eval_model(model, id_loader, device)
        gl_metrics = eval_model(model, gl_loader, device)
        gh_metrics = eval_model(model, gh_loader, device)

        key = 'canon_matrix_mse' if args.report_canonical else 'matrix_mse'
        print('epoch={:03d} train_loss={:.6f} id={:.6f} gl={:.6f} glhard={:.6f}'.format(
            epoch, train_loss, id_metrics[key], gl_metrics[key], gh_metrics[key]
        ))

        if gl_metrics[key] < best_gl:
            best_gl = gl_metrics[key]
            best_state = copy.deepcopy(model.state_dict())

    ckpt_path = os.path.join(args.output_dir, '{}_best.pt'.format(args.model))
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        model.load_state_dict(best_state)
        print('saved best checkpoint to {}'.format(ckpt_path))

    final = {
        'id': metric_subset(eval_model(model, id_loader, device)),
        'gl': metric_subset(eval_model(model, gl_loader, device)),
        'glhard': metric_subset(eval_model(model, gh_loader, device)),
        'scale': metric_subset(eval_model(model, sc_loader, device)),
        'shear': metric_subset(eval_model(model, sh_loader, device)),
    }
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='reln', choices=['avg_ls', 'noneq', 'ln', 'reln'])
    parser.add_argument('--train_transform', type=str, default='id', choices=['id', 'gl'])
    parser.add_argument('--train_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--num_traj', type=int, default=8)
    parser.add_argument('--traj_len', type=int, default=8)
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--ridge', type=float, default=1e-4)
    parser.add_argument('--stable_prob', type=float, default=0.7)
    parser.add_argument('--center_scale', type=float, default=0.35)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--exact_pair_eval', action='store_true')
    parser.add_argument('--report_canonical', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
