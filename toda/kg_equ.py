import torch
import numpy as np

def func(t, y):
    dH = np.zeros_like(y)
    N=32

    #dH[N]=y[1]-2*y[0]-np.sin(y[0])
    dH[N]=np.exp(y[N-1] - y[0]) - np.exp(y[0] - y[1])
    dH[0]=y[N]
    dH[2 * N - 1] = np.exp(y[N-2] - y[N-1]) - np.exp(y[N-1] - y[0])
    dH[N-1]=y[2*N-1]
    for i in range(1,N-1):
        dH[N+i]=np.exp(y[i-1]-y[i])-np.exp(y[i]-y[i+1])
        dH[i]=y[N+i]

    return dH

def func_N6(t, y):
    dH = np.zeros_like(y)
    N=6

    #dH[N]=y[1]-2*y[0]-np.sin(y[0])
    dH[N]=np.exp(y[N-1] - y[0]) - np.exp(y[0] - y[1])
    dH[0]=y[N]
    dH[2 * N - 1] = np.exp(y[N-2] - y[N-1]) - np.exp(y[N-1] - y[0])
    dH[N-1]=y[2*N-1]
    for i in range(1,N-1):
        dH[N+i]=np.exp(y[i-1]-y[i])-np.exp(y[i]-y[i+1])
        dH[i]=y[N+i]

    return dH

def func_N3(t, y):
    dH = np.zeros_like(y)
    N=3

    #dH[N]=y[1]-2*y[0]-np.sin(y[0])
    dH[N]=np.exp(y[N-1] - y[0]) - np.exp(y[0] - y[1])
    dH[0]=y[N]
    dH[2 * N - 1] = np.exp(y[N-2] - y[N-1]) - np.exp(y[N-1] - y[0])
    dH[N-1]=y[2*N-1]
    for i in range(1,N-1):
        dH[N+i]=np.exp(y[i-1]-y[i])-np.exp(y[i]-y[i+1])
        dH[i]=y[N+i]

    return dH