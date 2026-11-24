"""
WGANO for Elasticity (1D Model for Irregular Mesh)

Linear elasticity on irregular/unstructured meshes.
Uses 1D sequence model (Model1D) instead of 2D spatial model.

Data: Point cloud with 972 mesh points
- Input: (x, y) coordinates of each point
- Output: displacement (1 channel)

Optimal: n_hidden=144, n_layers=10, n_head=8, epochs=500, batch_size=8
"""
import os, sys, argparse, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import Model1D  # 1D model for irregular mesh
from utils import UnitGaussianNormalizer, TestLoss

torch.manual_seed(0); np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser('WGANO 1D for Elasticity')
    parser.add_argument('--n_hidden', type=int, default=144)
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='elasticity_wgano')
    return parser.parse_args()


def load_data(args):
    """
    Load elasticity data as 1D sequences.

    Expected format: [N, num_points, 3] where last dim is [x, y, displacement]
    Or separate files for coordinates and displacement.
    """
    data_path = os.path.join(args.data_dir, 'elasticity_data.npy')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = np.load(data_path)
    print(f"Loaded data shape: {data.shape}")

    # Assume format: [N, num_points, 3] -> [x, y, displacement]
    if len(data.shape) == 3 and data.shape[2] == 3:
        x_coords = torch.from_numpy(data[:, :, :2]).float()  # [N, num_points, 2]
        y_output = torch.from_numpy(data[:, :, 2:3]).float()  # [N, num_points, 1]
    else:
        # Try alternative format
        x_coords = torch.from_numpy(data[:args.ntrain+args.ntest, :, :2]).float()
        y_output = torch.from_numpy(data[:args.ntrain+args.ntest, :, 2:3]).float()

    num_points = x_coords.shape[1]

    # Split train/test
    x_train = x_coords[:args.ntrain]
    y_train = y_output[:args.ntrain]
    x_test = x_coords[args.ntrain:args.ntrain+args.ntest]
    y_test = y_output[args.ntrain:args.ntrain+args.ntest]

    return x_train, y_train, x_test, y_test, num_points


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("WGANO-1D for Elasticity (Irregular Mesh)")
    print("="*80)

    x_train, y_train, x_test, y_test, num_points = load_data(args)
    ntrain, ntest = len(x_train), len(x_test)
    print(f"Train: {ntrain}, Test: {ntest}, Points: {num_points}")

    # Normalize
    x_norm = UnitGaussianNormalizer(x_train.reshape(ntrain, -1))
    y_norm = UnitGaussianNormalizer(y_train.reshape(ntrain, -1))

    x_train = x_norm.encode(x_train.reshape(ntrain, -1)).reshape(ntrain, num_points, 2)
    x_test = x_norm.encode(x_test.reshape(ntest, -1)).reshape(ntest, num_points, 2)
    y_train = y_norm.encode(y_train.reshape(ntrain, -1)).reshape(ntrain, num_points, 1)
    y_test_tensor = y_test  # Keep original for evaluation

    train_loader = DataLoader(TensorDataset(x_train, y_train),
                             batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test_tensor),
                            batch_size=args.batch_size, shuffle=False)

    x_norm.cuda(); y_norm.cuda()

    # Create 1D model
    print("\nCreating Model1D (for irregular mesh)...")
    model = Model1D(
        in_dim=2,  # (x, y) coordinates
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        out_dim=1,
        mlp_ratio=1,
    ).cuda()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                   epochs=args.epochs,
                                                   steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    ckpt_path = os.path.join(args.save_dir, args.save_name + '.pt')
    best_error = float('inf')

    print("\nTraining...")
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)  # [B, num_points, 1]
            loss = myloss(y_norm.decode(out.reshape(-1, num_points)),
                         y_norm.decode(y.reshape(-1, num_points)))
            loss.backward(); optimizer.step(); scheduler.step()

        model.eval()
        test_error = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                test_error += myloss(y_norm.decode(out.reshape(-1, num_points)),
                                    y.reshape(-1, num_points)).item()
        test_error /= ntest

        if test_error < best_error:
            best_error = test_error
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Test: {test_error:.6f} | Best: {best_error:.6f}")

    print(f"\nBest error: {best_error:.6f}")


if __name__ == '__main__':
    main()
