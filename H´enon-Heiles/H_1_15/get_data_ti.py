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

sample=3000
N=2

u_0=io.loadmat('50_step_pred/IC.mat')['ic']

time=50
M=1001
t_eval = np.linspace(0, time, M) #[0,0.05,...,50]
flag = False
kwargs = {'rtol': 1e-10}
index=np.arange(0,M,20)#[0,20,...,1000]
y_inputs = [None] * 10 #长为10的列表
for i in range(sample):
    print('train', i)
    sol = solver(func, [0, time], u_0[i], t_eval=t_eval, **kwargs)
    tval = sol['t'][index]
    #[0,1,...,50]
    qp=sol['y'][:,index]
    ti_index=random.randint(0, 48)
    xval_ti = qp[:,ti_index]
    yval_next = [qp[:, ti_index + j + 1] for j in range(2)]


    if flag:
        x_input = np.vstack((x_input, xval_ti))
        for j in range(2):
            y_inputs[j] = np.vstack((y_inputs[j], yval_next[j]))
    else:
        x_input = xval_ti
        for j in range(2):
            y_inputs[j] = yval_next[j]
        flag = True

io.savemat('50_step_pred/data_ti.mat', {'data': x_input})
for j in range(2):
    filename = f'50_step_pred/data_next_{j+1}.mat'
    io.savemat(filename, {'data': y_inputs[j]})
