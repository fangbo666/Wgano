"""
C-WGANO for Magnetohydrodynamics (MHD)

MHD coupling: electromagnetic field ↔ velocity field
- Input: Br (magnetic field, 1 channel)
- Output 1: J (current density, 3 channels: Jx, Jy, Jz)
- Output 2: u (velocity, 2 channels: u, v)

Coupling mechanism: Lorentz force (J×B) and electromagnetic induction

Data format: .npy files with shape [H, W, 6]
  Channel 0: Br (input)
  Channels 1-3: Jx, Jy, Jz (EM output)
  Channels 4-5: u, v (Flow output)

Optimal hyperparameters:
- n_hidden: 64
- n_layers: 4
- n_head: 4
- head_hidden: 64
- coupling_iter: 2
- epochs: 500
- batch_size: 4
"""

import os
import sys
import argparse
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from wgano import MHDModel
from utils import TestLoss

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser('C-WGANO for MHD')

    # Model hyperparameters
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--head_hidden', type=int, default=64)
    parser.add_argument('--coupling_iter', type=int, default=2)

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)

    # Other
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='mhd_cwgano')
    parser.add_argument('--eval', action='store_true')

    return parser.parse_args()


def load_data(data_dir, split, n_samples):
    """Load MHD data from .npy files."""
    folder = os.path.join(data_dir, 'MHD', split)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")

    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')],
                   key=lambda x: int(os.path.splitext(x)[0]))

    data = np.stack([np.load(os.path.join(folder, f)) for f in files[:n_samples]])
    return data


def prepare_data(args):
    """Load and prepare MHD data."""
    H, W = 128, 128

    # Load data
    print("Loading data...")
    train_data = load_data(args.data_dir, 'train', args.ntrain)
    test_data = load_data(args.data_dir, 'test', args.ntest)

    # Extract channels
    x_train = torch.from_numpy(train_data[..., 0:1]).float()  # Br
    em_train = torch.from_numpy(train_data[..., 1:4]).float()  # Jx, Jy, Jz
    fl_train = torch.from_numpy(train_data[..., 4:6]).float()  # u, v

    x_test = torch.from_numpy(test_data[..., 0:1]).float()
    em_test = torch.from_numpy(test_data[..., 1:4]).float()
    fl_test = torch.from_numpy(test_data[..., 4:6]).float()

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(x_train, em_train, fl_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, em_test, fl_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def nrmse_per_channel(pred, target):
    """Compute per-channel NRMSE and return mean."""
    B, H, W, C = pred.shape
    errors = []
    for c in range(C):
        p = pred[..., c].reshape(B, -1)
        t = target[..., c].reshape(B, -1)
        err = (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).mean()
        errors.append(err)
    return sum(errors) / len(errors)


def train_epoch(model, train_loader, optimizer, scheduler, max_grad_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, em, fl in train_loader:
        x, em, fl = x.cuda(), em.cuda(), fl.cuda()

        optimizer.zero_grad()
        pred_em, pred_fl = model(x)

        # Compute per-channel NRMSE
        loss_em = nrmse_per_channel(pred_em, em)
        loss_fl = nrmse_per_channel(pred_fl, fl)
        loss = loss_em + loss_fl

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, test_loader):
    """Evaluate on test set."""
    model.eval()
    em_errors = []
    fl_errors = []

    with torch.no_grad():
        for x, em, fl in test_loader:
            x, em, fl = x.cuda(), em.cuda(), fl.cuda()
            pred_em, pred_fl = model(x)

            em_errors.append(nrmse_per_channel(pred_em, em).item())
            fl_errors.append(nrmse_per_channel(pred_fl, fl).item())

    return np.mean(em_errors), np.mean(fl_errors)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("C-WGANO for Magnetohydrodynamics (MHD)")
    print("="*80)

    # Prepare data
    train_loader, test_loader = prepare_data(args)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = MHDModel(
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_head=args.n_head,
        head_hidden=args.head_hidden,
        coupling_iter=args.coupling_iter,
        H=128,
        W=128,
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Save configuration
    config = {
        'task': 'MHD',
        'model': 'C-WGANO',
        'n_hidden': args.n_hidden,
        'n_layers': args.n_layers,
        'n_head': args.n_head,
        'head_hidden': args.head_hidden,
        'coupling_iter': args.coupling_iter,
        'total_params': total_params,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }

    with open(os.path.join(args.save_dir, args.save_name + '_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    ckpt_path = os.path.join(args.save_dir, args.save_name + '.pt')
    log_path = os.path.join(args.save_dir, args.save_name + '_log.csv')

    # Evaluation mode
    if args.eval:
        print("\nEvaluation mode...")
        model.load_state_dict(torch.load(ckpt_path))
        em_err, fl_err = evaluate(model, test_loader)
        print(f"EM error: {em_err:.6f}, Flow error: {fl_err:.6f}")
        return

    # Training loop
    print("\nStarting training...")
    print("-"*80)

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_em', 'test_flow', 'lr'])

    best_total = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                args.max_grad_norm)
        em_err, fl_err = evaluate(model, test_loader)
        total_err = em_err + fl_err

        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.6f} | "
              f"EM: {em_err:.6f} | "
              f"Flow: {fl_err:.6f} | "
              f"LR: {lr:.2e} | "
              f"Time: {elapsed:.0f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f'{train_loss:.6f}', f'{em_err:.6f}',
                f'{fl_err:.6f}', f'{lr:.2e}'
            ])

        if total_err < best_total:
            best_total = total_err
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved!")

    print("-"*80)
    print(f"\nTraining completed!")
    print(f"Best total error: {best_total:.6f}")
    print(f"Model saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
