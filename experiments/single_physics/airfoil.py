"""
WGANO for Airfoil Flow

Flow around NACA airfoil and cylinder geometries.
Special: Non-square grid (221x51), no normalization.

Data format:
- NACA_Cylinder_X.npy, NACA_Cylinder_Y.npy (coordinates)
- NACA_Cylinder_Q.npy (output, 5 channels, use channel 4)

Optimal: n_hidden=320, n_layers=10, n_head=8, epochs=800, batch_size=4
"""
import os, sys, argparse, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import Model
from utils import TestLoss

torch.manual_seed(0); np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hidden', type=int, default=320)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='airfoil_wgano')
    return parser.parse_args()


def load_data(args):
    """Load Airfoil data - non-square grid."""
    inputX = torch.from_numpy(np.load(os.path.join(args.data_dir, 'NACA_Cylinder_X.npy'))).float()
    inputY = torch.from_numpy(np.load(os.path.join(args.data_dir, 'NACA_Cylinder_Y.npy'))).float()
    inp = torch.stack([inputX, inputY], dim=-1)  # [N, 221, 51, 2]

    # Output: channel 4 (vorticity)
    output = torch.from_numpy(np.load(os.path.join(args.data_dir, 'NACA_Cylinder_Q.npy'))[:, 4]).float()

    print(f"Input shape: {inp.shape}, Output shape: {output.shape}")

    # Downsample
    r = args.downsample
    if r > 1:
        inp = inp[:, ::r, ::r]
        output = output[:, ::r, ::r]

    s1, s2 = inp.shape[1], inp.shape[2]  # Non-square: 221x51

    # Split
    x_train = inp[:args.ntrain]
    y_train = output[:args.ntrain].unsqueeze(-1)
    x_test = inp[args.ntrain:args.ntrain+args.ntest]
    y_test = output[args.ntrain:args.ntrain+args.ntest].unsqueeze(-1)

    return x_train, y_train, x_test, y_test, s1, s2


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("WGANO for Airfoil (Non-square grid)")
    x_train, y_train, x_test, y_test, s1, s2 = load_data(args)
    ntrain, ntest = len(x_train), len(x_test)
    print(f"Train: {ntrain}, Test: {ntest}, Grid: {s1}x{s2}")

    # No normalization for airfoil (use raw values)
    train_loader = DataLoader(TensorDataset(x_train, y_train),
                             batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test),
                            batch_size=args.batch_size, shuffle=False)

    # Model with non-square dimensions
    model = Model(
        space_dim=2, n_layers=args.n_layers, n_hidden=args.n_hidden,
        n_head=args.n_head, fun_dim=0, out_dim=1,
        unified_pos=False, H=s1, W=s2, use_qk_norm=True
    ).cuda()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                   epochs=args.epochs,
                                                   steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    ckpt_path = os.path.join(args.save_dir, args.save_name + '.pt')
    best_error = float('inf')

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out, _, _ = model(x, fx=None)
            loss = myloss(out, y)
            loss.backward(); optimizer.step(); scheduler.step()

        model.eval()
        test_error = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out, _, _ = model(x, fx=None)
                test_error += myloss(out, y).item()
        test_error /= ntest

        if test_error < best_error:
            best_error = test_error
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Test: {test_error:.6f} | Best: {best_error:.6f}")

    print(f"Best error: {best_error:.6f}")


if __name__ == '__main__':
    main()
