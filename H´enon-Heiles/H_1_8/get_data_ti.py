import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp
import random
from fun import *
import scipy.io as io
torch.backends.cudnn.deterministic = True

seed = 6
random.seed(seed)
np.random.seed(seed)

sample = 5000
N = 2
time = 50
M = 2 # 只修改这里：M=1、2、3或4
u_0 = io.loadmat('50_step_pred/IC.mat')['ic']
ti = np.random.uniform(0, time - M, sample)
flag = False
kwargs = {'rtol': 1e-10}
for i in range(sample):
    print('train', i)
    t_start = ti[i]
    t_points = np.array([t_start + j for j in range(M + 1)])
    sol = solver(func, [0, time], u_0[i], t_eval=t_points, **kwargs)
    qp = sol['y']
    xval_ti = qp[:, 0]
    next_values = [qp[:, j] for j in range(1, M + 1)]
    if flag:
        x_input = np.vstack((x_input, xval_ti))
        for j in range(M):
            data_next[j] = np.vstack((data_next[j], next_values[j]))
    else:
        x_input = xval_ti
        data_next = next_values.copy()
        flag = True
io.savemat('50_step_pred/data_ti.mat', {'data': x_input, 'ti': ti})
for j in range(M):
    io.savemat(f'50_step_pred/data_next_{j + 1}_M{M}.mat', {'data': data_next[j]})
print(f'M={M} data generation completed.')