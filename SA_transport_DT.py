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

# diffusive controlled transport 
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','D','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0, 0.0005], # lamb
               [0, 0.0005], # alpha
               [0, 0.0005]] # kd 
}

early_times = np.linspace(0,50000,10000)
late_times = np.linspace(50000,20000000,1000)

L = 2
x = 2
ts = 5
v = 0.001
Co = 1
param_values = saltelli.sample(problem, 2**10)
params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])
#%%

Y_early_dt = np.zeros(param_values.shape[0])
Y_peak_dt = np.zeros(param_values.shape[0])
Y_late_dt = np.zeros(param_values.shape[0])

# initialize btc list
btc_data = []
for i, X in tqdm(enumerate(param_values), desc='Running Analysis'):
    # run early time domain
    concentrations_early, adaptive_times_early = model.concentration_106_new_adaptive(early_times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    
    # run late time domain
    concentrations_late, adaptive_times_late = model.concentration_106_new_adaptive(late_times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)

    combined_times = np.concatenate([adaptive_times_early, adaptive_times_late])
    combined_concentrations = np.concatenate([concentrations_early, concentrations_late])

    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i] = model.calculate_metrics(combined_times, combined_concentrations)
    #print(f'Diffusive transport iteration: {i}')

    btc_data.append({
        "index":i,
        "params": X.tolist(),
        "times": combined_times.tolist(),
        "concentrations": combined_concentrations.tolist()
        })

# save metrics
metrics_df = pd.DataFrame({
    'Early': Y_early_dt,
    'Peak': Y_peak_dt,
    'Late': Y_late_dt
})

# Save metrics DataFrame
metrics_df.to_csv(os.path.join(output_dir, 'metrics_dt.csv'), index=True)

# Save BTC data as JSON
with open(os.path.join(output_dir, 'btc_data_dt.json'), 'w') as f:
    json.dump(btc_data, f)

# Sobol analysis and saving results
Si_early_dt = sobol.analyze(problem, Y_early_dt, print_to_console=False)
Si_peak_dt = sobol.analyze(problem, Y_peak_dt, print_to_console=False)
Si_late_dt = sobol.analyze(problem, Y_late_dt, print_to_console=False)

total_Si_early_dt, first_Si_early_dt, second_Si_early_dt = Si_early_dt.to_df()
total_Si_peak_dt, first_Si_peak_dt, second_Si_peak_dt = Si_peak_dt.to_df()
total_Si_late_dt, first_Si_late_dt, second_Si_late_dt = Si_late_dt.to_df()

# Save all Sobol indices results
total_Si_early_dt.to_csv(os.path.join(output_dir, 'total_Si_early_dt.csv'), index=True)
first_Si_early_dt.to_csv(os.path.join(output_dir, 'first_Si_early_dt.csv'), index=True)
second_Si_early_dt.to_csv(os.path.join(output_dir, 'second_Si_early_dt.csv'), index=True)
total_Si_peak_dt.to_csv(os.path.join(output_dir, 'total_Si_peak_dt.csv'), index=True)
first_Si_peak_dt.to_csv(os.path.join(output_dir, 'first_Si_peak_dt.csv'), index=True)
second_Si_peak_dt.to_csv(os.path.join(output_dir, 'second_Si_peak_dt.csv'), index=True)
total_Si_late_dt.to_csv(os.path.join(output_dir, 'total_Si_late_dt.csv'), index=True)
first_Si_late_dt.to_csv(os.path.join(output_dir, 'first_Si_late_dt.csv'), index=True)
second_Si_late_dt.to_csv(os.path.join(output_dir, 'second_Si_late_dt.csv'), index=True)