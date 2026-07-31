import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp #求解 ODE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from func_exp import *
import scipy.io as io
torch.backends.cudnn.determinstic = True
import math

seed=6
random.seed(seed)
np.random.seed(seed)

sample=3000
N=3
state_GRF=io.loadmat('state_u0_3000_lambdai_N3.mat')['state']
#[3000,6]
time=20
M=2001
t_eval = np.linspace(0, time, M) #[0,0.01,...,20]
index=np.arange(0,M,100)
flag = False
kwargs = {'rtol': 1e-12}
for i in range(sample):
    print('train', i)
    sol = solver(func_exp, [0, time], state_GRF[i], t_eval=t_eval, **kwargs)
    tval = sol['t'][index] #[0,1,...,20]
    qp=sol['y'][:,index]
    ti_index=random.randint(0, 18)
    xval_ti = qp[:,ti_index]
    yval_next = qp[:,ti_index+1]
    zval_next = qp[:, ti_index +2]
    if flag:
        x_input = np.vstack((x_input, xval_ti))
        y_input = np.vstack((y_input, yval_next))
        z_input = np.vstack((z_input, zval_next))
    else:
        x_input = xval_ti
        y_input = yval_next
        z_input = zval_next
        flag = True

io.savemat('data_ti_N3.mat',{'data':x_input})
io.savemat('data_next_1_N3.mat',{'data':y_input})
io.savemat('data_next_2_N3.mat',{'data':z_input})


