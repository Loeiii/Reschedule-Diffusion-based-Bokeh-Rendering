import numpy as np

def f(x, N, a, deta):
    return np.prod([(1 - x + i*deta/(N-1)) for i in range(N)]) - a

def df(x, N, deta):
    res = 0
    for i in range(N):
        tmp = 1
        for j in range(N):
            if j != i:
                tmp *= (1 - x + j*deta/(N-1))
        res += tmp
    return -res

def newton(n_sample, alpha, deta = 0.05, x0 = 0.1):
    xn = x0
    N = n_sample
    a = alpha
    deta = deta
    for n in range(50): #迭代10次
        xn = xn - f(xn, N, a, deta)/df(xn, N, deta)
    return xn