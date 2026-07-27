import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp #求解 ODE
import random
from fun import *
import scipy.io as io
torch.backends.cudnn.determinstic = True

seed=6
random.seed(seed)
np.random.seed(seed)
#
# sample=5000
# N=2
# u_0=io.loadmat('50_step_pred_5000/IC_H_1_8.mat')['ic']
# time=50
# M=1001
# index=np.arange(0,M,20) #[0,20,...,1000]
# t_eval = np.linspace(0, time, M) #[0,0.05,...,50]
# # print(t_eval)
# flag = False
# kwargs = {'rtol': 1e-10}
# muti_time=50
# output=np.zeros((muti_time,sample,2*N))
# all_time=1001
# groud_true=np.zeros((all_time,sample,2*N))
#
# for i in range(sample):
#     print('train', i)
#     sol = solver(func, [0, time], u_0[i], t_eval=t_eval, **kwargs)
#     groud_true[:, i, :] = sol['y'].T
#     for j in np.arange(muti_time):
#         tval = sol['t'][index]#print(tval) [0,1,2,...,100]
#         qp=sol['y'][:,index]
#         output[j,i,:]=qp[:,j+1]
#
# io.savemat('50_step_pred_5000/groud_true.mat',{'true':groud_true})
# for i in range(muti_time):
#     io.savemat('100_step_pred_5000/muti_time/u_{:.1f}.mat'.format(1*(i+1)),
#                         {'u{:.2f}'.format(1*(i+1)):output[i]})
#



import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


# ============================================================
# Load existing trajectory data
# shape: (1001, 5000, 4)
# ============================================================

data = io.loadmat(
    "50_step_pred_5000/groud_true.mat"
)

ground_true = data["true"]

print("Data shape:", ground_true.shape)


# ============================================================
# Extract q1, q2, p1, p2
# ============================================================

q1 = ground_true[:, :, 0]
q2 = ground_true[:, :, 1]
p1 = ground_true[:, :, 2]
p2 = ground_true[:, :, 3]


# ============================================================
# Compute Hamiltonian
#
# H = 1/2(p1^2+p2^2)
#   + 1/2(q1^2+q2^2)
#   + q1^2*q2
#   - 1/3*q2^3
#
# H_all shape: (1001, 5000)
# ============================================================

H_all = (
    0.5 * (p1 ** 2 + p2 ** 2)
    + 0.5 * (q1 ** 2 + q2 ** 2)
    + q1 ** 2 * q2
    - q2 ** 3 / 3.0
)


# ============================================================
# Energy drift
# ============================================================

# 每条轨迹自己的初始能量
H_initial = H_all[0, :]

# 相对于各自初始能量的绝对漂移
absolute_drift = np.abs(
    H_all - H_initial[None, :]
)

# 相对漂移
relative_drift = (
    absolute_drift
    / np.maximum(
        np.abs(H_initial[None, :]),
        1e-14,
    )
)

# 每条轨迹在[0,50]上的最大相对能量漂移
max_relative_drift_each = np.max(
    relative_drift,
    axis=0,
)


# ============================================================
# Statistics
# ============================================================

print("\nInitial energy:")
print("Target H =", 1.0 / 8.0)
print("Mean =", np.mean(H_initial))
print("Minimum =", np.min(H_initial))
print("Maximum =", np.max(H_initial))

print("\nEnergy conservation over t in [0,50]:")
print(
    "Mean maximum relative drift =",
    np.mean(max_relative_drift_each),
)

print(
    "Median maximum relative drift =",
    np.median(max_relative_drift_each),
)

print(
    "95% quantile =",
    np.percentile(max_relative_drift_each, 95),
)

print(
    "Largest relative drift =",
    np.max(max_relative_drift_each),
)

print(
    "Largest absolute drift =",
    np.max(absolute_drift),
)


# ============================================================
# Time-dependent statistics
# ============================================================

time = np.linspace(
    0.0,
    50.0,
    ground_true.shape[0],
)

mean_H = np.mean(H_all, axis=1)
std_H = np.std(H_all, axis=1)

mean_relative_drift = np.mean(
    relative_drift,
    axis=1,
)

median_relative_drift = np.median(
    relative_drift,
    axis=1,
)

percentile95_drift = np.percentile(
    relative_drift,
    95,
    axis=1,
)

maximum_relative_drift = np.max(
    relative_drift,
    axis=1,
)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.5),
)


# 所有5000条轨迹的能量
axes[0].plot(
    time,
    H_all,
    color="steelblue",
    linewidth=0.3,
    alpha=0.015,
)

axes[0].axhline(
    1.0 / 8.0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=r"$H=1/8$",
)

axes[0].plot(
    time,
    mean_H,
    color="red",
    linewidth=2,
    label="Mean",
)
axes[0].ticklabel_format(
    axis="y",
    style="plain",
    useOffset=False,
)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Hamiltonian")
axes[0].set_title("(a) Hamiltonian of all trajectories")
axes[0].legend()
axes[0].grid(alpha=0.25)


# 相对能量漂移
eps = 1e-18

axes[1].semilogy(
    time,
    np.maximum(median_relative_drift, eps),
    linewidth=1.8,
    label="Median",
)

axes[1].semilogy(
    time,
    np.maximum(mean_relative_drift, eps),
    linewidth=1.8,
    label="Mean",
)

axes[1].semilogy(
    time,
    np.maximum(percentile95_drift, eps),
    linewidth=1.8,
    label="95th percentile",
)

axes[1].semilogy(
    time,
    np.maximum(maximum_relative_drift, eps),
    linewidth=1.8,
    label="Maximum",
)

axes[1].set_xlabel("Time")
axes[1].set_ylabel("Relative energy drift")
axes[1].set_title("(b) Relative energy drift")
axes[1].legend()
axes[1].grid(
    alpha=0.25,
    which="both",
)

plt.tight_layout()

plt.savefig(
    "50_step_pred_5000/"
    "energy_conservation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Save calculated Hamiltonian values
# ============================================================

io.savemat(
    "50_step_pred_5000/"
    "energy_conservation.mat",
    {
        "time": time,
        "H_all": H_all,
        "H_initial": H_initial,
        "relative_drift": relative_drift,
        "max_relative_drift_each":
            max_relative_drift_each,
    },
)