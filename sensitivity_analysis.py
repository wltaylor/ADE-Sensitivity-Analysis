#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:28:56 2024

@author: williamtaylor
"""

import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpmath import invertlaplace, mp
mp.dps = 12
import model
from SALib.sample import saltelli
from SALib.analyze import sobol



problem = {
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.001, 2], # D
               [0.01, 1], # v
               [0, 0.5], # lamb
               [0, 1], # alpha
               [0, 1]] # kd
}

param_values = saltelli.sample(problem, 2**2)
times = np.linspace(0,12000,24000)
L = 2
Y_early = np.zeros([param_values.shape[0]])
Y_peak = np.zeros([param_values.shape[0]])
Y_late = np.zeros([param_values.shape[0]])

# evaluate the model at the sampled parameter values, across the same dimensionless time and fixed L
for i,X in enumerate(param_values):
    Y_early[i], Y_peak[i], Y_late[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    print(i)

# apply Sobol method to each set of results    
Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()


# save results
total_Si_early.to_csv('results/total_Si_early.csv', index=True)
first_Si_early.to_csv('results/first_Si_early.csv', index=True)
second_Si_early.to_csv('results/second_Si_early.csv', index=True)
total_Si_peak.to_csv('results/total_Si_peak.csv', index=True)
first_Si_peak.to_csv('results/first_Si_peak.csv', index=True)
second_Si_peak.to_csv('results/second_Si_peak.csv', index=True)
total_Si_late.to_csv('results/total_Si_late.csv', index=True)
first_Si_late.to_csv('results/first_Si_late.csv', index=True)
second_Si_late.to_csv('results/second_Si_late.csv', index=True)


