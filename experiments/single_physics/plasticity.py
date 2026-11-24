"""
WGANO for Plasticity (Time Series with Random Time Augmentation)

Material plasticity deformation over time.
Special features:
- Non-square grid: 101x31
- Time series: T=20 timesteps
- 4-channel output (deformation components)
- Random time permutation per sample (data augmentation)
- Timestep embedding

Optimal: n_hidden=240, n_layers=8, n_head=8, epochs=500, batch_size=8, use_qk_norm=False
"""
import os, sys, argparse, csv, json, time
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import Model, timestep_embedding
from utils import TestLoss

torch.manual_seed(0); np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser('WGANO for Plasticity')
    parser.add_argument('--n_hidden', type=int, default=240)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=0.1)

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=900)
    parser.add_argument('--ntest', type=int, default=80)
    parser.add_argument('--T', type=int, default=20, help='Number of timesteps')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='plasticity_wgano')
    return parser.parse_args()


def load_data(args):
    """Load plasticity data from .mat file."""
    data_path = os.path.join(args.data_dir, 'plas_N987_T20.mat')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = scio.loadmat(data_path)

    # Input: material properties [N, H, W]
    # Output: deformations [N, H, W, T, 4]
    input_data = torch.from_numpy(data['input']).float()
    output_data = torch.from_numpy(data['output']).float()

    print(f"Input shape: {input_data.shape}, Output shape: {output_data.shape}")

    s1, s2 = input_data.shape[1], input_data.shape[2]  # 101, 31
    T = output_data.shape[3]  # 20

    # Manually normalize input (zero-mean, unit-variance)
    input_mean = input_data.mean(dim=(1, 2), keepdim=True)
    input_std = input_data.std(dim=(1, 2), keepdim=True) + 1e-8
    input_normalized = (input_data - input_mean) / input_std

    # Split train/test
    x_train = input_normalized[:args.ntrain]
    y_train = output_data[:args.ntrain]
    x_test = input_normalized[args.ntrain:args.ntrain+args.ntest]
    y_test = output_data[args.ntrain:args.ntrain+args.ntest]

    return x_train, y_train, x_test, y_test, s1, s2, T


def random_time_collate(batch):
    """
    Custom collate function with per-sample random time permutation.

    This augments the data by randomly shuffling timesteps for each sample
    independently, making the model more robust to temporal ordering.
    """
    pos_list, fx_list, yy_list, perm_list = [], [], [], []

    for pos_i, fx_i, yy_i in batch:
        # Random permutation for this sample
        T = yy_i.shape[-2]
        perm = torch.randperm(T)

        pos_list.append(pos_i)
        fx_list.append(fx_i)
        yy_list.append(yy_i[..., perm, :])  # Permute time dimension
        perm_list.append(perm)

    return (torch.stack(pos_list), torch.stack(fx_list),
            torch.stack(yy_list), torch.stack(perm_list))


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("WGANO for Plasticity (Time Series)")
    print("="*80)

    # Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test, s1, s2, T = load_data(args)
    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    print(f"Train: {ntrain}, Test: {ntest}, Grid: {s1}x{s2}, T: {T}")

    # Create spatial grid
    gx = np.linspace(0, 1, s1)
    gy = np.linspace(0, 1, s2)
    xx, yy = np.meshgrid(gx, gy, indexing='ij')
    pos = torch.tensor(np.stack([xx, yy], axis=-1), dtype=torch.float)
    pos_train = pos.unsqueeze(0).repeat(ntrain, 1, 1, 1)
    pos_test = pos.unsqueeze(0).repeat(ntest, 1, 1, 1)

    # Reshape for time-series training
    x_train_expanded = x_train.unsqueeze(-1).repeat(1, 1, 1, 1)  # [N, H, W, 1]
    x_test_expanded = x_test.unsqueeze(-1).repeat(1, 1, 1, 1)

    # Data loaders with custom collate
    train_loader = DataLoader(
        TensorDataset(pos_train, x_train_expanded, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=random_time_collate
    )
    test_loader = DataLoader(
        TensorDataset(pos_test, x_test_expanded, y_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Create model
    print("\nCreating model...")
    model = Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=1,
        out_dim=4,  # 4-channel deformation
        unified_pos=False,
        H=s1,
        W=s2,
        use_qk_norm=False,  # Note: use_qk_norm=False for plasticity
    ).cuda()

    # Time embedding layer
    time_fc = nn.Linear(args.n_hidden, args.n_hidden).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(time_fc.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    myloss = TestLoss(size_average=False)

    ckpt_path = os.path.join(args.save_dir, args.save_name + '.pt')
    best_error = float('inf')

    print("\nStarting training...")
    print("-"*80)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0

        for pos, fx, yy, perm in train_loader:
            pos, fx, yy = pos.cuda(), fx.cuda(), yy.cuda()
            B, H, W = pos.shape[0], pos.shape[1], pos.shape[2]

            optimizer.zero_grad()

            # Process each timestep
            total_loss = 0
            for t in range(T):
                # Create timestep embedding
                t_tensor = torch.tensor([t / T], dtype=torch.float32).cuda()
                t_embed = timestep_embedding(t_tensor.unsqueeze(0).repeat(B, 1),
                                            args.n_hidden)
                t_embed = time_fc(t_embed).reshape(B, 1, 1, args.n_hidden)
                t_embed = t_embed.repeat(1, H, W, 1)

                # Forward pass
                out, _, _ = model(pos, fx=fx)  # [B, H, W, 4]
                target = yy[..., t, :]  # [B, H, W, 4]

                loss_t = myloss(out, target)
                total_loss = total_loss + loss_t

            loss = total_loss / T
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(time_fc.parameters()),
                    args.max_grad_norm
                )

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Evaluate
        model.eval()
        test_error = 0

        with torch.no_grad():
            for pos, fx, yy in test_loader:
                pos, fx, yy = pos.cuda(), fx.cuda(), yy.cuda()
                B, H, W = pos.shape[0], pos.shape[1], pos.shape[2]

                total_err = 0
                for t in range(T):
                    out, _, _ = model(pos, fx=fx)
                    target = yy[..., t, :]
                    total_err = total_err + myloss(out, target).item()

                test_error += total_err / T

        test_error /= ntest

        if test_error < best_error:
            best_error = test_error
            torch.save({
                'model': model.state_dict(),
                'time_fc': time_fc.state_dict()
            }, ckpt_path)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_loss/len(train_loader):.6f} | "
                  f"Test: {test_error:.6f} | "
                  f"Best: {best_error:.6f}")

    print("-"*80)
    print(f"Training completed! Best error: {best_error:.6f}")


if __name__ == '__main__':
    main()
