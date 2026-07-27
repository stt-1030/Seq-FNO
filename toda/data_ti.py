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
ti=np.random.uniform(0,18,sample)
time=20
flag = False
kwargs = {'rtol': 1e-12}
for i in range(sample):
    print('train', i)
    t_start = ti[i]
    t_points = np.array([t_start, t_start + 1, t_start + 2])
    sol = solver(func_N3, [0, time], state_GRF[i], t_eval=t_points, **kwargs)
    tval = sol['t']
    qp=sol['y']
    xval_ti = qp[:,0]
    yval_next = qp[:,1]
    zval_next = qp[:,2]
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


