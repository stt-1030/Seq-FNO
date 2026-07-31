import numpy as np

def func_exp(t, y):
    dH = np.zeros_like(y)
    dH[0]=-np.exp(-y[3])
    dH[1]=-np.exp(-y[4])
    dH[2]=-np.exp(-y[5])
    dH[3]=np.exp(y[2]-y[0])-np.exp(y[0]-y[1])
    dH[4]=np.exp(y[0]-y[1])-np.exp(y[1]-y[2])
    dH[5]=np.exp(y[1]-y[2])-np.exp(y[2]-y[0])
    return dH