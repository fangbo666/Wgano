"""
WGANO for Navier-Stokes Equation with Scheduled Sampling

2D incompressible Navier-Stokes with periodic boundary conditions.
Time-series prediction: use T_in=10 initial steps to predict next T=10 steps.

Special training: Scheduled Sampling
- Gradually increase probability of using model's own predictions
- Ramps from 0 to 0.9 between epochs 50-300
- Stabilizes autoregressive training

Optimal hyperparameters:
- n_hidden: 328, n_layers: 8, n_head: 8
- epochs: 500, batch_size: 2
- use_qk_norm: True
"""
import os, sys, argparse, csv, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import Model
from utils import UnitGaussianNormalizer, TestLoss

torch.manual_seed(0); np.random.seed(0)
torch.cuda.manual_seed(0); torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser('WGANO for Navier-Stokes')
    parser.add_argument('--n_hidden', type=int, default=328)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Scheduled sampling parameters
    parser.add_argument('--ss_start', type=int, default=50,
                       help='Epoch to start scheduled sampling ramp')
    parser.add_argument('--ss_end', type=int, default=300,
                       help='Epoch when SS reaches maximum')
    parser.add_argument('--ss_max', type=float, default=0.9,
                       help='Maximum SS probability')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ntrain', type=int, default=1000)
    parser.add_argument('--ntest', type=int, default=200)
    parser.add_argument('--T_in', type=int, default=10,
                       help='Number of input time steps')
    parser.add_argument('--T', type=int, default=10,
                       help='Number of output time steps to predict')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Spatial downsampling factor')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='ns_wgano')
    parser.add_argument('--eval', action='store_true')

    return parser.parse_args()


def get_ss_epsilon(epoch, args):
    """
    Compute scheduled sampling epsilon (probability of using model's own prediction).

    Linear ramp from 0 to ss_max between ss_start and ss_end epochs.
    """
    if epoch <= args.ss_start:
        return 0.0
    if epoch >= args.ss_end:
        return args.ss_max
    return args.ss_max * (epoch - args.ss_start) / (args.ss_end - args.ss_start)


def load_data(args):
    """Load Navier-Stokes data."""
    data_path = os.path.join(args.data_dir, 'ns_data_V1e-3_N5000_T50.npy')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path)  # [N, T, H, W]
    print(f"Loaded data shape: {data.shape}")

    # Downsample spatially if needed
    r = args.downsample
    if r > 1:
        data = data[:, :, ::r, ::r]

    s = data.shape[2]  # Spatial resolution

    # Split train/test
    x_train = torch.from_numpy(data[:args.ntrain, :args.T_in]).float()  # Initial states
    y_train = torch.from_numpy(data[:args.ntrain, args.T_in:args.T_in+args.T]).float()  # Future states

    x_test = torch.from_numpy(data[args.ntrain:args.ntrain+args.ntest, :args.T_in]).float()
    y_test = torch.from_numpy(data[args.ntrain:args.ntrain+args.ntest, args.T_in:args.T_in+args.T]).float()

    return x_train, y_train, x_test, y_test, s


def train_epoch_with_ss(model, train_loader, optimizer, scheduler, y_normalizer,
                        myloss, ss_epsilon, args):
    """
    Train one epoch with scheduled sampling.

    With probability ss_epsilon, use model's own prediction as input for next step
    instead of ground truth (teacher forcing).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    for pos, x_seq, y_seq in train_loader:
        # x_seq: [B, T_in, H, W], y_seq: [B, T, H, W]
        pos, x_seq, y_seq = pos.cuda(), x_seq.cuda(), y_seq.cuda()
        B, T_in, H, W = x_seq.shape
        T_out = y_seq.shape[1]

        optimizer.zero_grad()

        # Autoregressive prediction with scheduled sampling
        predictions = []
        current_input = x_seq[:, -1:, :, :]  # Last input frame [B, 1, H, W]

        for t in range(T_out):
            # Predict next frame
            fx = current_input.reshape(B, H, W, 1)
            out, _, _ = model(pos, fx=fx)  # [B, H, W, 1]
            pred = out.squeeze(-1)  # [B, H, W]
            predictions.append(pred)

            # Scheduled sampling: decide whether to use prediction or ground truth
            if t < T_out - 1:  # Not last step
                use_prediction = (torch.rand(1).item() < ss_epsilon)
                if use_prediction:
                    current_input = pred.unsqueeze(1)
                else:
                    current_input = y_seq[:, t:t+1, :, :]  # Ground truth

        # Stack predictions
        pred_seq = torch.stack(predictions, dim=1)  # [B, T, H, W]

        # Compute loss in normalized space
        loss = myloss(
            y_normalizer.encode(pred_seq.reshape(B*T_out, -1)).reshape(B, T_out, H, W),
            y_normalizer.encode(y_seq.reshape(B*T_out, -1)).reshape(B, T_out, H, W)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * B
        n_samples += B

    return total_loss / n_samples


def evaluate(model, test_loader, y_normalizer, myloss, args):
    """Evaluate on test set (fully autoregressive)."""
    model.eval()
    total_error = 0.0
    n_samples = 0

    with torch.no_grad():
        for pos, x_seq, y_seq in test_loader:
            pos, x_seq, y_seq = pos.cuda(), x_seq.cuda(), y_seq.cuda()
            B, T_in, H, W = x_seq.shape
            T_out = y_seq.shape[1]

            # Fully autoregressive prediction
            predictions = []
            current_input = x_seq[:, -1:, :, :]

            for t in range(T_out):
                fx = current_input.reshape(B, H, W, 1)
                out, _, _ = model(pos, fx=fx)
                pred = out.squeeze(-1)
                predictions.append(pred)
                current_input = pred.unsqueeze(1)  # Always use prediction

            pred_seq = torch.stack(predictions, dim=1)

            # Compute error in original space
            pred_denorm = y_normalizer.decode(pred_seq.reshape(B*T_out, -1)).reshape(B, T_out, H, W)
            error = myloss(pred_denorm, y_seq)

            total_error += error.item() * B
            n_samples += B

    return total_error / n_samples


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("WGANO for Navier-Stokes with Scheduled Sampling")
    print("="*80)

    # Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test, s = load_data(args)
    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    print(f"Train: {ntrain}, Test: {ntest}, Resolution: {s}x{s}")
    print(f"T_in: {args.T_in}, T_out: {args.T}")

    # Normalize (only on initial state, not time dimension)
    x_normalizer = UnitGaussianNormalizer(x_train.reshape(ntrain, -1))
    y_normalizer = UnitGaussianNormalizer(y_train.reshape(ntrain, -1))

    x_train = x_normalizer.encode(x_train.reshape(ntrain, -1)).reshape(ntrain, args.T_in, s, s)
    x_test = x_normalizer.encode(x_test.reshape(ntest, -1)).reshape(ntest, args.T_in, s, s)
    # Note: y is NOT encoded for training (loss computed after decoding)

    # Create coordinate grid
    gx, gy = torch.meshgrid(torch.linspace(0, 1, s), torch.linspace(0, 1, s), indexing='ij')
    pos = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(pos.repeat(ntrain, 1, 1, 1), x_train, y_train),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(pos.repeat(ntest, 1, 1, 1), x_test, y_test),
        batch_size=args.batch_size, shuffle=False
    )

    x_normalizer.cuda()
    y_normalizer.cuda()

    # Create model
    print("\nCreating model...")
    model = Model(
        space_dim=2,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        fun_dim=1,
        out_dim=1,
        unified_pos=True,
        ref=8,
        mlp_ratio=1,
        H=s,
        W=s,
        use_qk_norm=True,
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
    log_path = os.path.join(args.save_dir, args.save_name + '_log.csv')

    # Evaluation mode
    if args.eval:
        print("\nEvaluation mode...")
        model.load_state_dict(torch.load(ckpt_path))
        test_error = evaluate(model, test_loader, y_normalizer, myloss, args)
        print(f"Test error: {test_error:.6f}")
        return

    # Training loop
    print("\nStarting training with scheduled sampling...")
    print(f"SS ramp: epochs {args.ss_start}-{args.ss_end}, max={args.ss_max}")
    print("-"*80)

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_error', 'ss_epsilon', 'lr', 'time'])

    best_error = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        # Get scheduled sampling epsilon
        ss_eps = get_ss_epsilon(epoch, args)

        # Train
        train_loss = train_epoch_with_ss(
            model, train_loader, optimizer, scheduler,
            y_normalizer, myloss, ss_eps, args
        )

        # Evaluate
        test_error = evaluate(model, test_loader, y_normalizer, myloss, args)

        # Log
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.6f} | "
              f"Test: {test_error:.6f} | "
              f"SS_eps: {ss_eps:.3f} | "
              f"LR: {lr:.2e} | "
              f"Best: {best_error:.6f} | "
              f"Time: {elapsed:.0f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f'{train_loss:.6f}', f'{test_error:.6f}',
                f'{ss_eps:.3f}', f'{lr:.2e}', f'{elapsed:.1f}'
            ])

        # Save best model
        if test_error < best_error:
            best_error = test_error
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → New best model saved!")

    print("-"*80)
    print(f"\nTraining completed!")
    print(f"Best test error: {best_error:.6f}")


if __name__ == '__main__':
    main()
