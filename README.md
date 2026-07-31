# Seq-FNO

This repository contains the implementation of **Seq-FNO**, a sequence-to-sequence Fourier neural operator for learning long-term evolution operators from sparsely supervised trajectories.

The numerical examples include:

- Toda lattice;
- exponential Toda lattice;
- Hénon–Heiles system at \(H=1/20\), \(1/15\), \(1/10\), and \(1/8\).

## Requirements

Install the required packages:

```bash
pip install numpy scipy matplotlib h5py torch
```

The current scripts use `.cuda()` directly. A CUDA-enabled PyTorch installation and an NVIDIA GPU are therefore required.

Clone the repository:

```bash
git clone https://github.com/stt-1030/Seq-FNO.git
cd Seq-FNO
```

Run each script from its corresponding experiment directory because relative paths are used throughout the code.

## Repository Structure

```text
Seq-FNO/
├── toda/
├── toda_exp/
└── H´enon-Heiles/
    ├── H_1_20/
    ├── H_1_15/
    ├── H_1_10/
    └── H_1_8/
```

In each experiment:

- `data_ti.py` or `get_data_ti.py` generates sparse training data;
- `data_muti_time.py` or `get_data_muti_time.py` generates long-horizon reference data;
- `get_initial.py` generates Hénon–Heiles initial conditions;
- `Seq-FNO.py` trains the model and evaluates long-horizon predictions.

## Toda Lattice

The initial-condition file `state_u0_5000_new.mat` is included in `toda/`.

Run:

```bash
cd toda
python data_ti.py
python data_muti_time.py
python Seq-FNO.py
```

The scripts are executed in the following order:

1. `data_ti.py` generates the randomly sampled input states and two supervised future states.
2. `data_muti_time.py` generates the reference states from \(t=1\) to \(t=30\).
3. `Seq-FNO.py` trains Seq-FNO and performs long-horizon prediction.

The generated reference data are saved in:

```text
muti_time_1_N3/
```

The model, figures, and predictions are saved in:

```text
results/Seq-FNO/
```

## Exponential Toda Lattice

The initial-condition file `state_u0_3000_lambdai_N3.mat` is included in `toda_exp/`.

Run:

```bash
cd toda_exp
python data_ti.py
python data_muti_time.py
python Seq-FNO.py
```

The generated reference data are saved in:

```text
muti_time_1_N3/
```

The model, figures, and predictions are saved in:

```text
results/Seq-FNO/
```

## Hénon–Heiles System

The Hénon–Heiles experiments are provided at four energy levels:

| Directory | Energy |
| --- | --- |
| `H_1_20/` | \(H=1/20\) |
| `H_1_15/` | \(H=1/15\) |
| `H_1_10/` | \(H=1/10\) |
| `H_1_8/` | \(H=1/8\) |

To run an experiment, enter the corresponding directory. For example:

```bash
cd "H´enon-Heiles/H_1_10"
python get_initial.py
python get_data_ti.py
python get_data_muti_time.py
python Seq-FNO.py
```

Replace `H_1_10` with `H_1_20`, `H_1_15`, or `H_1_8` to run another energy level.

The scripts must be executed in this order:

1. `get_initial.py` generates initial conditions on the prescribed energy surface.
2. `get_data_ti.py` randomly selects a starting time for each trajectory and generates the supervised future states.
3. `get_data_muti_time.py` generates the reference states from \(t=1\) to \(t=50\).
4. `Seq-FNO.py` trains Seq-FNO and evaluates its long-horizon predictions.

The generated data are saved in:

```text
50_step_pred/
```

The model, figures, prediction data, and error results are saved in:

```text
results/Seq-FNO_M{M}/
```

## Number of Supervised Steps

For the Hénon–Heiles experiments, the number of supervised future steps is controlled by:

```python
M = 2
```

The supported values are:

```text
M = 1, 2, 3, or 4
```

The value of `M` must be the same in:

```text
get_data_ti.py
Seq-FNO.py
```

For example, to train with four supervised future states, set:

```python
M = 4
```

in both files, then run:

```bash
python get_data_ti.py
python get_data_muti_time.py
python Seq-FNO.py
```

`get_data_muti_time.py` only needs to be rerun when the initial conditions or reference trajectories are changed.

## Output Files

The main output directories are:

```text
model/    trained model
plot/     prediction figures
pred/     predicted trajectories
```

Prediction errors are saved as:

```text
pred_error_M{M}.mat
```

All generated datasets and predictions are stored in MATLAB `.mat` format.
