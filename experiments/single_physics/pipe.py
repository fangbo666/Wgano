"""
WGANO for Pipe Flow

Compressible Navier-Stokes in pipe with different Mach/Reynolds numbers.
Special: coordinates (X,Y) are inputs, not positional encoding.

Data format: Separate coordinate files
- Pipe_X.npy, Pipe_Y.npy (coordinates)
- Pipe_Q.npy (output flow field, channel 0 is density)

Optimal hyperparameters:
- n_hidden: 184, n_layers: 7, n_head: 8
- epochs: 500, batch_size: 4
- use_qk_norm: False
- unified_pos: False (coords are inputs)
- fun_dim: 0 (no auxiliary function)
"""
import os, sys, argparse, csv, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import Model
from utils import UnitTransformer, TestLoss

torch.manual_seed(0); np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser('WGANO for Pipe Flow')
    parser.add_argument('--n_hidden', type=int, default=184)
    parser.add_argument('--n_layers', type=int, default=7)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)
    parser.add_argument('--downsample', type=int, default=1)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='pipe_wgano')
    return parser.parse_args()


def load_data(args):
    """Load Pipe data from separate coordinate and output files."""
    # Load coordinates
    inputX = torch.from_numpy(np.load(os.path.join(args.data_dir, 'Pipe_X.npy'))).float()
    inputY = torch.from_numpy(np.load(os.path.join(args.data_dir, 'Pipe_Y.npy'))).float()
    inp = torch.stack([inputX, inputY], dim=-1)  # [N, H, W, 2]

    # Load output (density field, channel 0)
    output = torch.from_numpy(np.load(os.path.join(args.data_dir, 'Pipe_Q.npy'))[:, 0]).float()

    print(f"Input shape: {inp.shape}, Output shape: {output.shape}")

    # Downsample
    r = args.downsample
    if r > 1:
        inp = inp[:, ::r, ::r]
        output = output[:, ::r, ::r]

    s1, s2 = inp.shape[1], inp.shape[2]

    # Split train/test
    x_train = inp[:args.ntrain]
    y_train = output[:args.ntrain]
    x_test = inp[-args.ntest:]  # Take last ntest samples
    y_test = output[-args.ntest:]

    return x_train, y_train, x_test, y_test, s1, s2


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("WGANO for Pipe Flow")
    print("="*80)

    # Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test, s1, s2 = load_data(args)
    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    print(f"Train: {ntrain}, Test: {ntest}, Resolution: {s1}x{s2}")

    # Normalize coordinates and output
    x_normalizer = UnitTransformer(x_train.reshape(ntrain, -1, 2))
    y_normalizer = UnitTransformer(y_train.reshape(ntrain, -1))

    x_train = x_normalizer.encode(x_train.reshape(ntrain, -1, 2)).reshape(ntrain, s1, s2, 2)
    x_test = x_normalizer.encode(x_test.reshape(ntest, -1, 2)).reshape(ntest, s1, s2, 2)
    y_train = y_normalizer.encode(y_train.reshape(ntrain, -1)).reshape(ntrain, s1, s2, 1)
    y_test_tensor = y_test.reshape(ntest, s1, s2, 1)

    # Data loaders (coordinates are input, not positional encoding)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test_tensor),
        batch_size=args.batch_size, shuffle=False
    )

    x_normalizer.cuda()
    y_normalizer.cuda()

    # Create model
    print("\nCreating model...")
    model = Model(
        space_dim=2,        # Input is 2D coordinates
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=0,          # No auxiliary function
        out_dim=1,
        unified_pos=False,  # Coordinates are input, not positional encoding
        H=s1,
        W=s2,
        use_qk_norm=False,  # Note: use_qk_norm=False for pipe
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
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
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            # Forward: x is input (coordinates), no fx
            out, _, _ = model(x, fx=None)

            loss = myloss(
                y_normalizer.decode(out.squeeze(-1)),
                y_normalizer.decode(y.squeeze(-1))
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Evaluate
        model.eval()
        test_error = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out, _, _ = model(x, fx=None)
                test_error += myloss(
                    y_normalizer.decode(out.squeeze(-1)),
                    y.squeeze(-1)
                ).item()
        test_error /= ntest

        if test_error < best_error:
            best_error = test_error
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_loss/ntrain:.6f} | "
                  f"Test: {test_error:.6f} | "
                  f"Best: {best_error:.6f}")

    print("-"*80)
    print(f"Training completed! Best error: {best_error:.6f}")


if __name__ == '__main__':
    main()
