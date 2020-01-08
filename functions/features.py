import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import skew, kurtosis

# %% Funções
@jit(nopython=True)
def mpp(x):
    return np.array([np.mean(x[:, i]) for i in range(len(x[0]))])

def energy_(x):
    return np.sum(np.abs(x)**2)

def curtose(s, fs):
    m = np.zeros((len(s), s.shape[1]))
    for i in range(len(m)):
        m[i] = np.array([kurtosis(s[i, ii*fs:(ii+1)*fs])
                         for ii in range((len(s)//fs)-1)])
    return mpp(m)


def assimetria(s, fs):
    m = np.zeros((len(s), s.shape[1]))
    for i in range(len(m)):
        m[i] = np.array([skew(s[i, ii*fs:(ii+1)*fs])
                         for ii in range((len(s)//fs)-1)])
    return mpp(m)


def variancia(s, fs):
    m = np.zeros((len(s), s.shape[1]))
    for i in range(len(m)):
        m[i] = np.array([np.var(s[i, ii*fs:(ii+1)*fs])
                         for ii in range((len(s)//fs)-1)])
    return mpp(m)

def energia(s, fs):
    m = np.zeros((len(s), s.shape[1]))
    for i in range(len(m)):
        m[i] = energy([np.var(s[i, ii*fs:(ii+1)*fs])
                         for ii in range((len(s)//fs)-1)])
    return mpp(m)