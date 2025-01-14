import numpy as np
import pandas as pd
from scipy import special
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
from tqdm import tqdm
import os

output_dir = '/Users/williamtaylor/Documents/GitHub/ADE-Sensitivity-Analysis/results'
os.makedirs(output_dir, exist_ok=True)

# advective controlled transport 
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','D','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0, 0.05], # lamb
               [0, 0.05], # alpha
               [0, 0.05]] # kd
}
times = np.linspace(0,500,1000)
L = 2
x = 2
ts = 5
v = 0.5
Co = 1

param_values = saltelli.sample(problem, 2**12)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])

Y_early_at = np.zeros(param_values.shape[0])
Y_peak_at = np.zeros(param_values.shape[0])
Y_late_at = np.zeros(param_values.shape[0])

btc_data = []

for i, X in tqdm(enumerate(param_values), desc='Running Analysis'):
    concentrations, adaptive_times = model.concentration_106_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)

    Y_early_at[i], Y_peak_at[i], Y_late_at[i] = model.calculate_metrics(adaptive_times, concentrations)

    btc_data.append({
        "index": i,
        "params": X.tolist(),
        "times": adaptive_times,
        "concentrations": concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early_at,
    'Peak': Y_peak_at,
    'Late': Y_late_at
})

# Save metrics DataFrame
metrics_df.to_csv(os.path.join(output_dir, 'metrics_at.csv'), index=True)

# Save BTC data as JSON
with open(os.path.join(output_dir, 'btc_data_at.json'), 'w') as f:
    json.dump(btc_data, f)

# Sobol analysis and saving results
Si_early_at = sobol.analyze(problem, Y_early_at, print_to_console=False)
Si_peak_at = sobol.analyze(problem, Y_peak_at, print_to_console=False)
Si_late_at = sobol.analyze(problem, Y_late_at, print_to_console=False)

total_Si_early_at, first_Si_early_at, second_Si_early_at = Si_early_at.to_df()
total_Si_peak_at, first_Si_peak_at, second_Si_peak_at = Si_peak_at.to_df()
total_Si_late_at, first_Si_late_at, second_Si_late_at = Si_late_at.to_df()

# Save all Sobol indices results
total_Si_early_at.to_csv(os.path.join(output_dir, 'total_Si_early_at.csv'), index=True)
first_Si_early_at.to_csv(os.path.join(output_dir, 'first_Si_early_at.csv'), index=True)
second_Si_early_at.to_csv(os.path.join(output_dir, 'second_Si_early_at.csv'), index=True)
total_Si_peak_at.to_csv(os.path.join(output_dir, 'total_Si_peak_at.csv'), index=True)
first_Si_peak_at.to_csv(os.path.join(output_dir, 'first_Si_peak_at.csv'), index=True)
second_Si_peak_at.to_csv(os.path.join(output_dir, 'second_Si_peak_at.csv'), index=True)
total_Si_late_at.to_csv(os.path.join(output_dir, 'total_Si_late_at.csv'), index=True)
first_Si_late_at.to_csv(os.path.join(output_dir, 'first_Si_late_at.csv'), index=True)
second_Si_late_at.to_csv(os.path.join(output_dir, 'second_Si_late_at.csv'), index=True)