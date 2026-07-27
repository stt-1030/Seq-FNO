import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp #求解 ODE
import random
from fun import *
import scipy.io as io
torch.backends.cudnn.determinstic = True
import argparse
import math


seed=6
random.seed(seed)
np.random.seed(seed)

sample=3000
N=2
u_0=io.loadmat('50_step_pred/IC.mat')['ic']
time=100
M=1001
index=np.arange(0,M,20) #[0,20,...,1000]
t_eval = np.linspace(0, time, M) #[0,0.05,...,50]
# print(t_eval)
flag = False
kwargs = {'rtol': 1e-10}
muti_time=50
output=np.zeros((muti_time,sample,2*N))
all_time=1001
groud_true=np.zeros((all_time,sample,2*N))

for i in range(sample):
    print('train', i)
    sol = solver(func, [0, time], u_0[i], t_eval=t_eval, **kwargs)
    groud_true[:, i, :] = sol['y'].T
    for j in np.arange(muti_time):
        tval = sol['t'][index]
        qp=sol['y'][:,index]
        output[j,i,:]=qp[:,j+1]

io.savemat('50_step_pred/groud_true.mat', {'true':groud_true})
for i in range(muti_time):
    io.savemat('50_step_pred/muti_time/u_{:.1f}.mat'.format(1*(i+1)),
                        {'u{:.2f}'.format(1*(i+1)):output[i]})




