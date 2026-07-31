import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp #求解 ODE
import random
from kg_equ import *
import scipy.io as io
torch.backends.cudnn.determinstic = True

seed=6
random.seed(seed)
np.random.seed(seed)

sample=5000
N=3
state_GRF=io.loadmat('state_u0_5000_new.mat')['state']
time=30
M=3001
t_eval = np.linspace(0, time, M)
index=np.arange(0,M,100)
flag = False
kwargs = {'rtol': 1e-12}
muti_time=30
output=np.zeros((muti_time,sample,2*N))
for i in range(sample):
    print('train', i)
    sol = solver(func_N3, [0, time], state_GRF[i], t_eval=t_eval, **kwargs)
    for j in np.arange(muti_time):
        tval = sol['t'][index]
        qp=sol['y'][:,index] #(6,21)
        output[j,i,:]=qp[:,j+1]


for i in range(muti_time):
    io.savemat('muti_time_1_N3/u_GRF_lambdai_{:.1f}.mat'.format(1*(i+1)),
                        {'u{:.2f}'.format(1*(i+1)):output[i]})




