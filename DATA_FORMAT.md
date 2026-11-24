# Data Format Specification

This document describes the expected data format for each task.

---

## Single-Physics Tasks

### 1. Darcy Flow

**Data files:**
```
data/darcy/
├── piececonst_r421_N1024_smooth1.mat  (training)
└── piececonst_r421_N1024_smooth2.mat  (testing, optional)
```

**Format:**
- MATLAB `.mat` files
- Fields: `coeff` (input, permeability), `sol` (output, pressure)
- Shape: `[N, 421, 421]`
- Downsampled to 85x85 (downsample factor = 5)

**Preprocessing:**
- Normalize with `UnitTransformer`

---

### 2. Navier-Stokes (NS)

**Data files:**
```
data/ns/
└── ns_data_V1e-3_N5000_T50.npy
```

**Format:**
- NumPy `.npy` file
- Shape: `[N, T, H, W]` where T=50 time steps
- Uses first 10 steps (T_in=10) to predict next 10 steps

**Special requirements:**
- Time-series autoregressive training
- Scheduled sampling: ramp from 0 to 0.9 between epochs 50-300

**Preprocessing:**
- Normalize with `UnitGaussianNormalizer`

---

### 3. Pipe Flow

**Data files:**
```
data/pipe/
├── Pipe_X.npy  (x-coordinates)
├── Pipe_Y.npy  (y-coordinates)
└── Pipe_Q.npy  (output flow field)
```

**Format:**
- Separate coordinate and output files
- Shape: `[N, 129, 129]`
- Input: stacked (X, Y) coordinates
- Output: Q (density field, channel 0)

**Special settings:**
- `unified_pos=False` (coordinates are input, not positional encoding)
- `fun_dim=0` (no auxiliary function input)

**Preprocessing:**
- Normalize with `UnitTransformer`

---

### 4. Airfoil

**Data files:**
```
data/airfoil/
├── NACA_Cylinder_X.npy  (x-coordinates)
├── NACA_Cylinder_Y.npy  (y-coordinates)
└── NACA_Cylinder_Q.npy  (output fields, 5 channels)
```

**Format:**
- **Non-square grid:** `[N, 221, 51]`
- Input: stacked (X, Y) coordinates
- Output: Q[:, 4] (5th channel - vorticity)

**Special settings:**
- `unified_pos=False`
- **No normalization** (use raw values)
- Non-square spatial dimensions

---

### 5. Elasticity (Irregular Mesh)

**Data files:**
```
data/elasticity/
└── elasticity_data.npy
```

**Format:**
- **1D sequence data** (irregular mesh flattened)
- Shape: `[N, 972, 2]` where 972 = number of mesh points
- Input: (x, y) coordinates of each point
- Output: displacement (1 channel)

**Special requirements:**
- Uses **1D model** (`Model1D`)
- Processes as sequence, not 2D grid
- `DWT_1D` instead of `DWT_2D`

**Preprocessing:**
- Normalize with `UnitGaussianNormalizer`

---

### 6. Plasticity

**Data files:**
```
data/plasticity/
└── plas_N987_T20.mat
```

**Format:**
- MATLAB `.mat` file
- Fields: `input` (material properties), `output` (deformations)
- **Non-square grid:** `[N, 101, 31]`
- **Time series:** T=20 time steps
- Output: 4-channel deformation field

**Special settings:**
- `unified_pos=False`
- `Time_Input=True` (requires timestep embedding)
- Custom time augmentation: random permutation per sample

**Preprocessing:**
- Manual normalization: zero-mean, unit-variance on input
- No normalization on output

---

## Multi-Physics Tasks

All multi-physics tasks use similar data format:

```
data/
├── MHD/
│   ├── train/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   └── test/
│       └── ...
├── E_flow/
│   ├── train/
│   └── test/
├── NS_heat/
│   ├── train/
│   └── test/
└── VA/
    ├── train/
    └── test/
```

### 7. MHD (Magnetohydrodynamics)

**Format:**
- Shape: `[128, 128, 6]`
- Channel 0: Br (input magnetic field)
- Channels 1-3: Jx, Jy, Jz (output current density)
- Channels 4-5: u, v (output velocity)

**Model:** `MHDModel` (3 EM channels + 2 flow channels)

---

### 8. E-flow (Electroosmotic Flow)

**Format:**
- Shape: `[128, 128, 4]`
- Channel 0: conductivity (input)
- Channel 1: potential (output)
- Channels 2-3: u, v (output velocity)

**Model:** `EFlowModel` (1 potential + 2 velocity channels)

---

### 9. NS-heat (Thermal Convection)

**Format:**
- Shape: `[128, 128, 4]`
- Channel 0: heat source (input)
- Channels 1-2: u, v (output velocity)
- Channel 3: temperature (output)

**Model:** `NSHeatModel` (2 velocity + 1 temperature channels)

---

### 10. VA (Vibroacoustics)

**Format:**
- Shape: `[128, 128, 13]`
- Channel 0: density (input)
- Channels 1-6: acoustic field (3 complex = 6 real, output)
- Channels 7-12: structural field (3 complex = 6 real, output)

**Model:** `VAModel` (6 acoustic + 6 structural channels)

---

## Data Availability

### Option 1: Use Provided Datasets

Download from Zenodo:
```
https://doi.org/10.5281/zenodo.24132237
```

Contains: MHD, E_flow, NS_heat, VA (multi-physics benchmarks)

### Option 2: Generate Your Own

For single-physics tasks, refer to:
- PDEBench: https://github.com/pdebench/PDEBench
- Original papers for specific parameters

---

## Quick Reference Table

| Task | Grid Size | Input Channels | Output Channels | Model Type | Normalizer |
|------|-----------|----------------|-----------------|------------|------------|
| Darcy | 85×85 | 1 | 1 | 2D | UnitTransformer |
| NS | 64×64 | 10 (time) | 1 | 2D + temporal | UnitGaussianNormalizer |
| Pipe | 128×128 | 2 (coords) | 1 | 2D | UnitTransformer |
| Airfoil | 221×51 | 2 (coords) | 1 | 2D | None |
| Elasticity | 972 pts | 2 | 1 | **1D** | UnitGaussianNormalizer |
| Plasticity | 101×31 | 1 + time | 4 | 2D + temporal | Manual |
| MHD | 128×128 | 1 | 3+2 | C-WGANO | None |
| E-flow | 128×128 | 1 | 1+2 | C-WGANO | None |
| NS-heat | 128×128 | 1 | 2+1 | C-WGANO | None |
| VA | 128×128 | 1 | 6+6 | C-WGANO | None |

---

## Notes

1. **Resolution:** Most tasks can work with different resolutions, but model `H` and `W` parameters must match data.

2. **Batch size:** Adjust based on GPU memory. Recommended values are in experiment scripts.

3. **Data preparation:** If your data format differs, modify the `load_data()` function in experiment scripts.

4. **Irregular meshes:** Elasticity requires special handling as point cloud, not regular grid.

5. **Time series:** NS and Plasticity require special temporal processing logic.
