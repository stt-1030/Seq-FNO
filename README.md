# Seq-FNO

This repository contains the implementation of **Seq-FNO**, a sequence-to-sequence Fourier neural operator for learning long-term evolution laws from sparsely supervised trajectories. The code includes experiments for the Toda lattice, the exponential Toda lattice, and the Hénon–Heiles Hamiltonian system at four energy levels.

## Repository Structure

| Directory | System | Main files |
| --- | --- | --- |
| `toda/` | Toda lattice | `data_ti.py`, `data_muti_time.py`, `Seq-FNO.py` |
| `toda_exp/` | Exponential Toda lattice | `data_ti.py`, `data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_20/` | Hénon–Heiles, \(H=1/20\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_15/` | Hénon–Heiles, \(H=1/15\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_10/` | Hénon–Heiles, \(H=1/10\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |
| `H´enon-Heiles/H_1_8/` | Hénon–Heiles, \(H=1/8\) | `get_initial.py`, `get_data_ti.py`, `get_data_muti_time.py`, `Seq-FNO.py` |

The remaining files provide the supporting components:

- `Adam.py`: optimizer implementation;
- `utilities3.py`: data-loading and loss utilities;
- `kg_equ.py`: governing equations of the Toda lattice;
- `func_exp.py`: governing equations of the exponential Toda lattice;
- `fun.py`: governing equations of the Hénon–Heiles system.

## Requirements

The code requires Python and the following packages:

```bash
pip install numpy scipy matplotlib h5py torch
