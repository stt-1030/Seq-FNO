# Seq-FNO

This repository contains the implementation of **Seq-FNO**, a sequence-to-sequence Fourier neural operator for learning long-term evolution laws from sparsely supervised trajectories. The code includes experiments for the Toda lattice, an exponential-interaction Toda system, and the Hénon–Heiles Hamiltonian system at four energy levels.

## Repository structure

| Directory | System | Main files |
| --- | --- | --- |
| `toda/` | Toda lattice | `data_ti.py`, `data_muti_time.py`, `Seq-FNO.py` |
| `toda_exp/` | Exponential-interaction Toda system | `data_ti.py`, `data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_20/` | Hénon–Heiles, \(H=1/20\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_15/` | Hénon–Heiles, \(H=1/15\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_10/` | Hénon–Heiles, \(H=1/10\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_8/` | Hénon–Heiles, \(H=1/8\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |

`Adam.py` provides the optimizer implementation, `utilities3.py` contains data-loading and loss utilities, and `kg_equ.py`, `func_exp.py`, and `fun.py` define the corresponding dynamical systems.

## Requirements

The code is written in Python and requires the following packages:

```bash
pip install numpy scipy matplotlib h5py torch
```

The training and prediction scripts call `.cuda()` directly, so a CUDA-enabled PyTorch installation and an NVIDIA GPU are required unless the device handling in the scripts is changed.

## Data-generation workflow

Run each command from the corresponding experiment directory because the scripts use relative paths.

### Toda lattice

The initial-condition file `state_u0_5000_new.mat` is included in `toda/`. Generate the sparse training pairs and the long-horizon reference trajectories before running Seq-FNO:

```bash
cd toda
python data_ti.py
mkdir -p muti_time_1_N50
python data_muti_time.py
python Seq-FNO.py
```

The scripts have the following roles:

1. `data_ti.py` randomly selects one state from each trajectory and generates its next two states. It writes:
   - `data_ti_N3.mat`: input states;
   - `data_next_1_N3.mat`: first future states;
   - `data_next_2_N3.mat`: second future states.
2. `data_muti_time.py` generates the reference trajectories used for long-horizon prediction and error evaluation.
3. `Seq-FNO.py` trains or loads Seq-FNO, produces future predictions, and evaluates the prediction errors.

> **Current-code note:** in `toda/Seq-FNO.py`, the training block and `torch.save(...)` line are commented out, and the script directly loads `results/FNO_N3_3/model/FNO_N3_3`. Uncomment the training block to train from scratch. The evaluation code also contains one reference to `muti_time_1_N3`; change it to `muti_time_1_N50` to match the output directory of `data_muti_time.py`.

### Exponential-interaction Toda system

The initial-condition file `state_u0_3000_lambdai_N3.mat` is included in `toda_exp/`. Run:

```bash
cd toda_exp
python data_ti.py
mkdir -p muti_time_1_N50
python data_muti_time.py
python Seq-FNO.py
```

Here, `data_ti.py` generates the sparse input and two supervised future states, while `data_muti_time.py` generates the long-horizon reference data. `Seq-FNO.py` then trains the model and evaluates the predicted trajectories.

> **Current-code note:** the evaluation code in `toda_exp/Seq-FNO.py` contains one reference to `muti_time_1_N3`; change it to `muti_time_1_N50` to match the generated reference-data directory.

### Hénon–Heiles system

For \(H=1/20\), \(1/15\), and \(1/10\), enter the selected energy directory and run:

```bash
cd "H´enon-Heiles/H_1_10"   # replace H_1_10 with H_1_20 or H_1_15 if needed
mkdir -p 50_step_pred/muti_time
python get_initial.py
python get_data_ti.py
python get_data_muti_time.py
python Seq-FNO.py
```

For \(H=1/8\), the scripts use a different data-directory name:

```bash
cd "H´enon-Heiles/H_1_8"
mkdir -p 50_step_pred_5000/muti_time 100_step_pred_5000/muti_time
python get_initial.py
python get_data_ti.py
python get_data_muti_time.py
python Seq-FNO.py
```

The Hénon–Heiles files are used in the following order:

1. `get_initial.py` generates initial conditions on the prescribed energy surface.
2. `get_data_ti.py` integrates the system, randomly selects a starting state from each trajectory, and saves two supervised future states.
3. `get_data_muti_time.py` generates the full reference trajectories or processes them for long-horizon evaluation.
4. `Seq-FNO.py` trains or loads the model and saves predictions under `results/FNO_3/pred/`.

The principal MATLAB files are:

| File | MATLAB key | Description |
| --- | --- | --- |
| `IC.mat` or `IC_H_1_8.mat` | `ic` | Initial states \([q_1,q_2,p_1,p_2]\) |
| `data_ti.mat` | `data` | Randomly selected input states |
| `data_next_1.mat` | `data` | First supervised future states |
| `data_next_2.mat` | `data` | Second supervised future states |
| `groud_true.mat` | `true` | Full reference trajectories |
| `muti_time/u_*.mat` | `u*` | Reference states at individual prediction times |

> **Current-code notes**
>
> - In `H_1_20/get_data_muti_time.py`, the loop that writes `muti_time/u_*.mat` is commented out. Uncomment it before long-horizon evaluation with `Seq-FNO.py`.
> - In `H_1_20`, `H_1_15`, and `H_1_10`, the training block in `Seq-FNO.py` is commented out and the script directly loads `results/FNO_3/model/FNO_3`. Uncomment the training block and `torch.save(...)` to train from scratch.
> - In `H_1_8/get_data_muti_time.py`, the trajectory-generation section is currently commented out. Enable that section to generate `groud_true.mat` and the time-indexed reference files before running a complete experiment from scratch.
> - Check that the `50_step_pred_5000` and `100_step_pred_5000` references in the \(H=1/8\) scripts match the prediction horizon you intend to reproduce.

## Training configuration

The main hyperparameters are defined directly in each `Seq-FNO.py` file:

- `ntrain` and `ntest`: numbers of training and test trajectories;
- `batch_size`: mini-batch size;
- `epochs`: number of training epochs;
- `modes`: number of retained Fourier modes;
- `width`: latent channel width;
- `learning_rate`, `step_size`, and `gamma`: optimizer and learning-rate scheduler settings.

Modify these values in the corresponding script before training if a different configuration is required.

## Reproducibility notes

- NumPy, Python, and PyTorch random seeds are set in the scripts.
- The provided code stores data and predictions in MATLAB `.mat` format.
- Generated datasets and model checkpoints can be large and are therefore not all included in the repository.
- Directory and file names such as `muti_time` and `groud_true.mat` are retained to match the paths used by the source code.
