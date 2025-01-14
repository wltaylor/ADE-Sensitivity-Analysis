#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:00:28 2024

@author: williamtaylor
"""

import numpy as np
import pandas as pd
from mpmath import invertlaplace, mp
mp.dps = 12
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
from tqdm import tqdm
import os

output_dir = '/Users/williamtaylor/Documents/GitHub/ADE-Sensitivity-Analysis/results'
os.makedirs(output_dir, exist_ok=True)

problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','D','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.001, 2], # D
               [0, 0.5], # lamb
               [0, 1], # alpha
               [0, 1]] # kd
}

param_values = saltelli.sample(problem, 2**10)
times = np.linspace(0,18000,3000)

# static parameters, consistent between both models
Co = 1
L = 2
x=2
ts=5
v=0.1

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# evaluate model 106
Y_early = np.zeros([param_values.shape[0]])
Y_peak = np.zeros([param_values.shape[0]])
Y_late = np.zeros([param_values.shape[0]])

# initiaze a list for the btcs
btc_data = []

# evaluate the model at the sampled parameter values, across the same dimensionless time and fixed L
for i,X in tqdm(enumerate(param_values), desc='Running Analysis Model 106'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],Co=Co, v=v, ts=ts, L=L, x=x)

    Y_early[i], Y_peak[i], Y_late[i] = model.calculate_metrics(adaptive_times, concentrations)

    btc_data.append({
        'index':i,
        'params':X.tolist(),
        'times':adaptive_times,
        'concentrations':concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early,
    'Peak': Y_peak,
    'Late': Y_late
})

metrics_df.to_csv(os.path.join(output_dir, 'metrics_106.csv'), index=True)

# save BTCs to json
with open(os.path.join(output_dir, 'btc_data_model106.json'), 'w') as f:
    json.dump(btc_data, f)

# apply Sobol method to each set of results    
Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

# save results
total_Si_early.to_csv(os.path.join(output_dir, 'total_Si_early_106.csv'), index=True)
first_Si_early.to_csv(os.path.join(output_dir, 'first_Si_early_106.csv'), index=True)
second_Si_early.to_csv(os.path.join(output_dir, 'second_Si_early_106.csv'), index=True)
total_Si_peak.to_csv(os.path.join(output_dir, 'total_Si_peak_106.csv'), index=True)
first_Si_peak.to_csv(os.path.join(output_dir, 'first_Si_peak_106.csv'), index=True)
second_Si_peak.to_csv(os.path.join(output_dir, 'second_Si_peak_106.csv'), index=True)
total_Si_late.to_csv(os.path.join(output_dir, 'total_Si_late_106.csv'), index=True)
first_Si_late.to_csv(os.path.join(output_dir, 'first_Si_late_106.csv'), index=True)
second_Si_late.to_csv(os.path.join(output_dir, 'second_Si_late_106.csv'), index=True)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# evaluate model 102 with the same parameters

Y_early = np.zeros([param_values.shape[0]])
Y_peak = np.zeros([param_values.shape[0]])
Y_late = np.zeros([param_values.shape[0]])

# initiaze a list for the btcs
btc_data = []

# evaluate the model at the sampled parameter values, across the same dimensionless time and fixed L
for i,X in tqdm(enumerate(param_values), desc='Running Analysis Model 102'):
    concentrations, adaptive_times = model.concentration_102_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)

    Y_early[i], Y_peak[i], Y_late[i] = model.calculate_metrics(adaptive_times, concentrations)

    btc_data.append({
        'index':i,
        'params':X.tolist(),
        'times':adaptive_times,
        'concentrations':concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early,
    'Peak': Y_peak,
    'Late': Y_late
})

metrics_df.to_csv(os.path.join(output_dir, 'metrics_102.csv'), index=True)

# save BTCs to json
with open(os.path.join(output_dir, 'btc_data_model102.json'), 'w') as f:
    json.dump(btc_data, f)

# apply Sobol method to each set of results    
Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

# save results
total_Si_early.to_csv(os.path.join(output_dir, 'total_Si_early_102.csv'), index=True)
first_Si_early.to_csv(os.path.join(output_dir, 'first_Si_early_102.csv'), index=True)
second_Si_early.to_csv(os.path.join(output_dir, 'second_Si_early_102.csv'), index=True)
total_Si_peak.to_csv(os.path.join(output_dir, 'total_Si_peak_102.csv'), index=True)
first_Si_peak.to_csv(os.path.join(output_dir, 'first_Si_peak_102.csv'), index=True)
second_Si_peak.to_csv(os.path.join(output_dir, 'second_Si_peak_102.csv'), index=True)
total_Si_late.to_csv(os.path.join(output_dir, 'total_Si_late_102.csv'), index=True)
first_Si_late.to_csv(os.path.join(output_dir, 'first_Si_late_102.csv'), index=True)
second_Si_late.to_csv(os.path.join(output_dir, 'second_Si_late_102.csv'), index=True)
