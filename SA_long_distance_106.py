#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:34 2024

@author: williamtaylor
"""
import os
os.chdir('/Users/williamtaylor/Documents/GitHub/ADE-Sensitivity-Analysis')
import numpy as np
import pandas as pd
from scipy import special
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
from tqdm import tqdm

output_dir = '/Users/williamtaylor/Documents/GitHub/ADE-Sensitivity-Analysis/results'
os.makedirs(output_dir, exist_ok=True)

# set parameters
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','D','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0, 1], # lamb
               [0, 1], # alpha
               [0, 1]] # kd 
}

ts = 750
x = 300
L = 300
v = 0.1
times = np.linspace(0,75000,1000)
param_values = saltelli.sample(problem, 2**1)
params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])

Y_early = np.zeros(param_values.shape[0])
Y_peak = np.zeros(param_values.shape[0])
Y_late = np.zeros(param_values.shape[0])

btc_data = []

for i, X in tqdm(enumerate(param_values), desc='Running Analysis'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],Co=1, v=v, ts=ts, L=L, x=x)

    Y_early[i], Y_peak[i], Y_late[i] = model.calculate_metrics(adaptive_times, concentrations)

    btc_data.append({
        'index':i,
        'params':X.tolist(),
        'times':adaptive_times,
        'concentrations':concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early':Y_early,
    'Peak':Y_peak,
    'Late':Y_late
    })

metrics_df.to_csv(os.path.join(output_dir, 'metrics_long_106.csv'), index=True)

# save btcs to json
with open(os.path.join(output_dir, 'btc_data_long_model106.json'), 'w') as f:
    json.dump(btc_data, f)

Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

total_Si_early.to_csv('results/spatial/total_Si_early_106.csv', index=True)
first_Si_early.to_csv('results/spatial/first_Si_early_106.csv', index=True)
second_Si_early.to_csv('results/spatial/second_Si_early_106.csv', index=True)
total_Si_peak.to_csv('results/spatial/total_Si_peak_106.csv', index=True)
first_Si_peak.to_csv('results/spatial/first_Si_peak_106.csv', index=True)
second_Si_peak.to_csv('results/spatial/second_Si_peak_106.csv', index=True)
total_Si_late.to_csv('results/spatial/total_Si_late_106.csv', index=True)
first_Si_late.to_csv('results/spatial/first_Si_late_106.csv', index=True)
second_Si_late.to_csv('results/spatial/second_Si_late_106.csv', index=True)

