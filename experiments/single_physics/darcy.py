"""
WGANO for Darcy Flow

Darcy flow equation with variable permeability coefficient.
Task: Given permeability field (input), predict pressure field (output).

Data format: .mat files with 'coeff' (input) and 'sol' (output)
Resolution: 421x421 → downsampled to 85x85

Optimal hyperparameters:
- n_hidden: 156
- n_layers: 9
- n_head: 6
- epochs: 650
- batch_size: 8
- use_qk_norm: True
"""

import os
import sys
import argparse
import csv
import json
import time
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from wgano import Model
from utils import UnitTransformer, TestLoss

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser('WGANO for Darcy Flow')

    # Model hyperparameters
    parser.add_argument('--n_hidden', type=int, default=156,
                        help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=9,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=int, default=2,
                        help='MLP expansion ratio')
    parser.add_argument('--use_qk_norm', action='store_true', default=True,
                        help='Use QK normalization (n1qk variant)')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=650,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=3e-5,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=0.1,
                        help='Max gradient norm for clipping')
    parser.add_argument('--pct_start', type=float, default=0.25,
                        help='OneCycleLR pct_start')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing .mat files')
    parser.add_argument('--ntrain', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--ntest', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--downsample', type=int, default=5,
                        help='Downsampling factor (421 → 85)')

    # Other
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_name', type=str, default='darcy_wgano',
                        help='Checkpoint name prefix')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluation mode')

    return parser.parse_args()


def load_data(args):
    """Load Darcy flow data from .mat files."""
    train_path = os.path.join(args.data_dir, 'piececonst_r421_N1024_smooth1.mat')
    test_path = os.path.join(args.data_dir, 'piececonst_r421_N1024_smooth2.mat')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    # Load training data
    train_data = scio.loadmat(train_path)
    r = args.downsample
    s = int(((421 - 1) / r) + 1)  # 85

    x_train = torch.from_numpy(
        train_data['coeff'][:args.ntrain, ::r, ::r][:, :s, :s]
    ).float()
    y_train = torch.from_numpy(
        train_data['sol'][:args.ntrain, ::r, ::r][:, :s, :s]
    ).float()

    # Load test data
    if os.path.exists(test_path):
        test_data = scio.loadmat(test_path)
        x_test = torch.from_numpy(
            test_data['coeff'][:args.ntest, ::r, ::r][:, :s, :s]
        ).float()
        y_test = torch.from_numpy(
            test_data['sol'][:args.ntest, ::r, ::r][:, :s, :s]
        ).float()
    else:
        # Use end of training data if test file doesn't exist
        x_test = torch.from_numpy(
            train_data['coeff'][args.ntrain:args.ntrain+args.ntest, ::r, ::r][:, :s, :s]
        ).float()
        y_test = torch.from_numpy(
            train_data['sol'][args.ntrain:args.ntrain+args.ntest, ::r, ::r][:, :s, :s]
        ).float()

    return x_train, y_train, x_test, y_test, s


def prepare_data(x_train, y_train, x_test, y_test, s, batch_size):
    """Normalize data and create data loaders."""
    ntrain, ntest = x_train.shape[0], x_test.shape[0]

    # Normalize
    x_normalizer = UnitTransformer(x_train.reshape(ntrain, -1))
    y_normalizer = UnitTransformer(y_train.reshape(ntrain, -1))

    x_train = x_normalizer.encode(x_train.reshape(ntrain, -1)).reshape(ntrain, s, s, 1)
    x_test = x_normalizer.encode(x_test.reshape(ntest, -1)).reshape(ntest, s, s, 1)
    y_train = y_normalizer.encode(y_train.reshape(ntrain, -1)).reshape(ntrain, s, s, 1)
    y_test = y_test.reshape(ntest, s, s, 1)

    # Create coordinate grid
    gx, gy = torch.meshgrid(torch.linspace(0, 1, s), torch.linspace(0, 1, s), indexing='ij')
    pos = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, s, s, 2]

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(pos.repeat(ntrain, 1, 1, 1), x_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(pos.repeat(ntest, 1, 1, 1), x_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, x_normalizer, y_normalizer


def train_epoch(model, train_loader, optimizer, scheduler, y_normalizer, myloss, max_grad_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for pos, fx, y in train_loader:
        pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()

        optimizer.zero_grad()
        out, _, _ = model(pos, fx=fx)

        # Compute loss in original space
        loss = myloss(
            y_normalizer.decode(out.squeeze(-1)),
            y_normalizer.decode(y.squeeze(-1))
        )

        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_samples += pos.shape[0]

    return total_loss / n_samples


def evaluate(model, test_loader, y_normalizer, myloss):
    """Evaluate on test set."""
    model.eval()
    total_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for pos, fx, y in test_loader:
            pos, fx, y = pos.cuda(), fx.cuda(), y.cuda()
            out, _, _ = model(pos, fx=fx)
            error = myloss(y_normalizer.decode(out.squeeze(-1)), y.squeeze(-1))
            total_error += error.item()
            n_samples += pos.shape[0]

    return total_error / n_samples


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("WGANO for Darcy Flow")
    print("="*80)

    # Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test, s = load_data(args)
    print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    print(f"Resolution: {s}x{s}")

    # Prepare data
    train_loader, test_loader, x_norm, y_norm = prepare_data(
        x_train, y_train, x_test, y_test, s, args.batch_size
    )
    x_norm.cuda()
    y_norm.cuda()

    # Create model
    print("\nCreating model...")
    model = Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        mlp_ratio=args.mlp_ratio,
        fun_dim=1,
        out_dim=1,
        unified_pos=True,
        ref=8,
        H=s,
        W=s,
        use_qk_norm=args.use_qk_norm,
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"use_qk_norm: {args.use_qk_norm}")

    # Save configuration
    config = {
        'task': 'Darcy Flow',
        'model': 'WGANO',
        'n_hidden': args.n_hidden,
        'n_layers': args.n_layers,
        'n_head': args.n_head,
        'mlp_ratio': args.mlp_ratio,
        'use_qk_norm': args.use_qk_norm,
        'total_params': total_params,
        'ntrain': args.ntrain,
        'ntest': args.ntest,
        'resolution': f'{s}x{s}',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
    }

    config_path = os.path.join(args.save_dir, args.save_name + '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start
    )
    myloss = TestLoss(size_average=False)

    ckpt_path = os.path.join(args.save_dir, args.save_name + '.pt')
    log_path = os.path.join(args.save_dir, args.save_name + '_log.csv')

    # Evaluation mode
    if args.eval:
        print("\nEvaluation mode...")
        model.load_state_dict(torch.load(ckpt_path))
        test_error = evaluate(model, test_loader, y_norm, myloss)
        print(f"Test relative L2 error: {test_error:.6f}")
        return

    # Training loop
    print("\nStarting training...")
    print("-"*80)

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_error', 'lr', 'time'])

    best_error = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            y_norm, myloss, args.max_grad_norm
        )

        # Evaluate
        test_error = evaluate(model, test_loader, y_norm, myloss)

        # Log
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.6f} | "
              f"Test: {test_error:.6f} | "
              f"LR: {lr:.2e} | "
              f"Best: {best_error:.6f} | "
              f"Time: {elapsed:.0f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f'{train_loss:.6f}', f'{test_error:.6f}',
                f'{lr:.2e}', f'{elapsed:.1f}'
            ])

        # Save best model
        if test_error < best_error:
            best_error = test_error
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved!")

    print("-"*80)
    print(f"\nTraining completed!")
    print(f"Best test error: {best_error:.6f}")
    print(f"Model saved to: {ckpt_path}")


if __name__ == '__main__':
    main()
