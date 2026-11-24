"""
C-WGANO for Electroosmotic Flow (E-flow)
Coupling: electric potential ↔ velocity field
Input: conductivity (1 channel)
Output 1: potential (1 channel)
Output 2: velocity (2 channels: u, v)
"""
import os, sys, argparse, csv, json, time, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from wgano import EFlowModel

torch.manual_seed(0); np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--head_hidden', type=int, default=64)
    parser.add_argument('--coupling_iter', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_name', type=str, default='eflow_cwgano')
    return parser.parse_args()

def load_data(data_dir, split, n_samples):
    folder = os.path.join(data_dir, 'E_flow', split)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    data = np.stack([np.load(os.path.join(folder, f)) for f in files[:n_samples]])
    return data

def nrmse_per_channel(pred, target):
    B, H, W, C = pred.shape
    errors = []
    for c in range(C):
        p = pred[..., c].reshape(B, -1)
        t = target[..., c].reshape(B, -1)
        err = (torch.norm(p - t, dim=1) / (torch.norm(t, dim=1) + 1e-8)).mean()
        errors.append(err)
    return sum(errors) / len(errors)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.save_dir, exist_ok=True)

    train_data = load_data(args.data_dir, 'train', 1000)
    test_data = load_data(args.data_dir, 'test', 200)

    x_train = torch.from_numpy(train_data[..., 0:1]).float()
    pot_train = torch.from_numpy(train_data[..., 1:2]).float()
    vel_train = torch.from_numpy(train_data[..., 2:4]).float()

    x_test = torch.from_numpy(test_data[..., 0:1]).float()
    pot_test = torch.from_numpy(test_data[..., 1:2]).float()
    vel_test = torch.from_numpy(test_data[..., 2:4]).float()

    train_loader = DataLoader(TensorDataset(x_train, pot_train, vel_train),
                             batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, pot_test, vel_test),
                            batch_size=args.batch_size, shuffle=False)

    model = EFlowModel(n_hidden=args.n_hidden, n_layers=args.n_layers,
                      n_head=args.n_head, head_hidden=args.head_hidden,
                      coupling_iter=args.coupling_iter).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                   epochs=args.epochs,
                                                   steps_per_epoch=len(train_loader),
                                                   pct_start=0.1)

    ckpt_path = os.path.join(args.save_dir, args.save_name + '.pt')
    best_total = float('inf')

    for epoch in range(args.epochs):
        model.train()
        for x, pot, vel in train_loader:
            x, pot, vel = x.cuda(), pot.cuda(), vel.cuda()
            optimizer.zero_grad()
            pred_pot, pred_vel = model(x)
            loss = nrmse_per_channel(pred_pot, pot) + nrmse_per_channel(pred_vel, vel)
            loss.backward(); optimizer.step(); scheduler.step()

        model.eval()
        pot_errors, vel_errors = [], []
        with torch.no_grad():
            for x, pot, vel in test_loader:
                x, pot, vel = x.cuda(), pot.cuda(), vel.cuda()
                pred_pot, pred_vel = model(x)
                pot_errors.append(nrmse_per_channel(pred_pot, pot).item())
                vel_errors.append(nrmse_per_channel(pred_vel, vel).item())

        pot_err, vel_err = np.mean(pot_errors), np.mean(vel_errors)
        total_err = pot_err + vel_err

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Pot: {pot_err:.6f} | Vel: {vel_err:.6f}")

        if total_err < best_total:
            best_total = total_err
            torch.save(model.state_dict(), ckpt_path)

    print(f"Best total error: {best_total:.6f}")

if __name__ == '__main__':
    main()
