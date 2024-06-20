from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import invertlaplace, mp

mp.dps = 12

def C_laplace(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2):
    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    
    # Here we use mp.sqrt and mp.exp from mpmath for multiprecision calculations
    r1 = 1 / (2 * D) * (v + mp.sqrt(v ** 2 + 4 * D * big_theta))
    r2 = 1 / (2 * D) * (v - mp.sqrt(v ** 2 + 4 * D * big_theta))
    
    # Convert the arguments to mpmath types before calling exp
    term1_numerator = r2 * mp.exp(r2 * L + r1 * x) - r1 * mp.exp(r1 * L + r2 * x)
    term1_denominator = r2 * mp.exp(r2 * L) - r1 * mp.exp(r1 * L)
    
    # Make sure the division is done using mpmath's div function for multiprecision calculations
    term1 = mp.fdiv(term1_numerator, term1_denominator)
    
    # Division should be handled by mpmath to maintain compatibility
    C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
    return C

def concentration_at_time(t, theta, rho_b, D, v, lamb, alpha, kd, Co):
    concentration = [invertlaplace(lambda s: C_laplace(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog') for time in t]
    print('transformed')
    return concentration

problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.000001, 1], # D
               [0.0001, 1], # v
               [0, 1], # lamb
               [0, 1], # alpha
               [0, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**6)


times = np.linspace(1,30,30)
y = np.array([concentration_at_time(times, *params) for params in param_values])
Si = [sobol.analyze(problem, Y) for Y in y.T]
#%%
S1s = np.array([s['S1'] for s in Si])
S2s = np.array([s['S2'] for s in Si])
STs = np.array([s['ST'] for s in Si])


for i in range(0,8):
    
    plt.plot(S1s[:,i], label = problem['names'][i])
plt.legend()
plt.show()

for i in range(0,8):
    
    plt.plot(S2s[:,i], label = problem['names'][i])
plt.legend()
plt.show()

for i in range(0,8):
    
    plt.plot(STs[:,i], label = problem['names'][i])
plt.legend()
plt.show()