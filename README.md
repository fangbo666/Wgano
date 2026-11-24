# WGANO: Wavelet-Guided Attention Neural Operator

Official PyTorch implementation of **WGANO** for solving single-physics and multi-physics PDEs.

**Author:** Bo Fang  
**Institution:** China University of Petroleum (East China)

---

## 📚 Overview

WGANO introduces a novel **high-frequency guided wavelet attention** mechanism that:
- Decomposes features into low-frequency and high-frequency components via Discrete Wavelet Transform (DWT)
- Uses HF components to guide attention on LF features through a gating mechanism
- Achieves state-of-the-art performance on multiple PDE benchmarks

**C-WGANO** extends this to multi-physics problems with:
- Physics-coupled bidirectional gating between multiple fields
- Iterative refinement with shared parameters
- Efficient handling of complex field interactions

---

## 🔧 Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```
wgano_clean/
├── wgano/
│   ├── __init__.py
│   ├── model.py              # WGANO model (single-physics)
│   └── coupled_model.py      # C-WGANO model (multi-physics)
├── utils/
│   ├── __init__.py
│   ├── normalizer.py         # Data normalization
│   └── testloss.py           # Relative L2 error
├── experiments/
│   ├── single_physics/       # 6 single-physics tasks
│   │   ├── darcy.py
│   │   ├── ns.py
│   │   ├── pipe.py
│   │   ├── airfoil.py
│   │   ├── elasticity.py
│   │   └── plasticity.py
│   └── multi_physics/        # 4 multi-physics tasks
│       ├── mhd.py
│       ├── eflow.py
│       ├── nsheat.py
│       └── va.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Single-Physics Example (Darcy Flow)

```python
from wgano import Model
import torch

# Create model
model = Model(
    space_dim=2,
    n_layers=9,
    n_hidden=156,
    n_head=6,
    fun_dim=1,
    out_dim=1,
    H=128,
    W=128,
    use_qk_norm=True,  # Set to False for n1 variant
)

# Forward pass
x = torch.randn(8, 128, 128, 2)  # Batch of coordinates
fx = torch.randn(8, 128, 128, 1)  # Input function
out, _, _ = model(x, fx=fx)
print(out.shape)  # [8, 128, 128, 1]
```

### Multi-Physics Example (MHD)

```python
from wgano import MHDModel

# Create C-WGANO for MHD
model = MHDModel(
    n_hidden=64,
    n_layers=4,
    n_head=4,
    head_hidden=64,
    coupling_iter=2,
    H=128,
    W=128,
)

# Forward pass
x = torch.randn(8, 128, 128, 3)  # Input with Br channel
em_pred, flow_pred = model(x)
print(em_pred.shape, flow_pred.shape)  # [8,128,128,3], [8,128,128,2]
```

---

## 📊 Experiments

### Single-Physics Tasks

Train on Darcy Flow:
```bash
python experiments/single_physics/darcy.py \
    --data_dir /path/to/data \
    --n_hidden 156 \
    --n_layers 9 \
    --n_head 6 \
    --epochs 650 \
    --batch_size 8 \
    --lr 1e-3
```

### Multi-Physics Tasks

Train on MHD:
```bash
python experiments/multi_physics/mhd.py \
    --data_dir /path/to/data \
    --n_hidden 64 \
    --n_layers 4 \
    --n_head 4 \
    --head_hidden 64 \
    --epochs 500 \
    --batch_size 4
```

---

## 🎯 Model Variants

### WGANO Variants

| Variant | `use_qk_norm` | Description |
|---------|---------------|-------------|
| **n1** | `False` | Base WGANO model |
| **n1qk** | `True` | With QK LayerNorm (default, better performance) |

### Hyperparameters

Optimal configurations from experiments:

| Task       | n_hidden | n_layers | n_head | epochs | batch_size | use_qk_norm |
|------------|----------|----------|--------|--------|------------|-------------|
| Darcy      | 156      | 9        | 6      | 650    | 8          | True        |
| NS         | 328      | 8        | 8      | 500    | 2          | True        |
| Pipe       | 184      | 7        | 8      | 500    | 4          | False       |
| Airfoil    | 320      | 10       | 8      | 800    | 4          | True        |
| Elasticity | 144      | 10       | 8      | 500    | 8          | False       |
| Plasticity | 240      | 8        | 8      | 500    | 8          | False       |

Multi-physics (C-WGANO): `n_hidden=64, n_layers=4, head_hidden=64, coupling_iter=2`

---

## 📖 Documentation

### Model API

#### WGANO (Single-Physics)

```python
Model(
    space_dim=2,        # Spatial dimension (2 for 2D PDEs)
    n_layers=4,         # Number of transformer blocks
    n_hidden=64,        # Hidden dimension
    n_head=4,           # Number of attention heads
    fun_dim=0,          # Input function dimension (0 if no auxiliary input)
    out_dim=1,          # Output dimension
    unified_pos=True,   # Use unified positional encoding
    ref=8,              # Reference scale for positional encoding
    mlp_ratio=1,        # MLP expansion ratio
    H=128,              # Height resolution
    W=128,              # Width resolution
    use_qk_norm=True,   # Use QK normalization (n1qk variant)
)
```

#### C-WGANO (Multi-Physics)

```python
DualPhysicsCoupledModel(
    n_hidden=64,         # Backbone hidden dimension
    n_layers=4,          # Number of transformer layers
    n_head=4,            # Number of attention heads
    head_hidden=64,      # Task-specific head dimension
    out_dim_1=3,         # Output dimension of field 1
    out_dim_2=2,         # Output dimension of field 2
    coupling_iter=2,     # Number of coupling iterations (K)
    H=128,               # Height resolution
    W=128,               # Width resolution
    use_qk_norm=True,    # Use QK normalization
)
```

Or use task-specific models: `MHDModel`, `EFlowModel`, `NSHeatModel`, `VAModel`

---

## 📦 Data Format

### Single-Physics

Expected data structure:
```
data/
├── Darcy/
│   ├── train/
│   │   ├── 0.npy  # [H, W, 2] - [input, output]
│   │   ├── 1.npy
│   │   └── ...
│   └── test/
│       └── ...
```

Each `.npy` file: `[H, W, channels]`
- Input channel(s): boundary conditions, parameters, etc.
- Output channel(s): solution field(s)

### Multi-Physics

```
data/
├── MHD/
│   ├── train/
│   │   ├── 0.npy  # [H, W, 6] - [Br, Jx, Jy, Jz, u, v]
│   │   └── ...
│   └── test/
│       └── ...
```

Channel layout varies by problem - see experiment scripts for details.

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{fang2026wgano,
  author       = {Fang, Bo},
  title        = {WGANO: Wavelet-Guided Attention Neural Operator},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.24344060},
  url          = {https://github.com/fangbo666/WGANO-official}
}
```

Dataset:
```bibtex
@dataset{fang2026wgano_data,
  author       = {Fang, Bo},
  title        = {WGANO Multi-Physics Benchmarks},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24132237}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

**Bo Fang**  
Email: s25070074@s.upc.edu.cn  
Institution: China University of Petroleum (East China)

For questions or issues:
- Open an issue on GitHub
- Contact via email

---

## 🙏 Acknowledgments

This work is based on research in neural operators and wavelet-based attention mechanisms.
Special thanks to the MultiphysicsBench dataset for providing the multi-physics benchmarks.
