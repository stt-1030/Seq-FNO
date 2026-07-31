import numpy as np
def func(t, y):
    dH = np.zeros_like(y)
    dH[0]=y[2]
    dH[1]=y[3]
    dH[2]=-y[0]-2*y[0]*y[1]
    dH[3]=-y[1]-y[0]**2+y[1]**2
    return dH


