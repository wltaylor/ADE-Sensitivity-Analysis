import numpy as np
import pandas as pd
from scipy import special
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
import json
#%%
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
param_values = saltelli.sample(problem, 2**1)
params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])
#%%

Y_early_dt = np.zeros(param_values.shape[0])
Y_peak_dt = np.zeros(param_values.shape[0])
Y_late_dt = np.zeros(param_values.shape[0])

# initialize btc list
btc_data = []
for i, X in enumerate(param_values):
    # run early time domain, don't record the late time tailing metric
    _, _, _, concentrations_early, adaptive_times_early = model.concentration_106_new_adaptive(early_times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    
    # run late time domain, don't record the early and peak metrics
    _, _, _, concentrations_late, adaptive_times_late = model.concentration_106_new_adaptive(late_times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)

    combined_times = np.concatenate([adaptive_times_early, adaptive_times_late])
    combined_concentrations = np.concatenate([concentrations_early, concentrations_late])

    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i] = model.calculate_metrics(combined_times, combined_concentrations)
    print(Y_late_dt[i])
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
metrics_df.to_csv('results/metrics_dt.csv', index=True)

# save BTCs to json
with open('results/btc_data_dt.json', 'w') as f:
    json.dump(btc_data, f)

# perform sobol analysis
Si_early_dt = sobol.analyze(problem, Y_early_dt, print_to_console=False)
Si_peak_dt = sobol.analyze(problem, Y_peak_dt, print_to_console=False)
Si_late_dt = sobol.analyze(problem, Y_late_dt, print_to_console=False)

total_Si_early_dt, first_Si_early_dt, second_Si_early_dt = Si_early_dt.to_df()
total_Si_peak_dt, first_Si_peak_dt, second_Si_peak_dt = Si_peak_dt.to_df()
total_Si_late_dt, first_Si_late_dt, second_Si_late_dt = Si_late_dt.to_df()

# save sensitivity results
total_Si_early_dt.to_csv('results/total_Si_early_dt.csv', index=True)
first_Si_early_dt.to_csv('results/first_Si_early_dt.csv', index=True)
second_Si_early_dt.to_csv('results/second_Si_early_dt.csv', index=True)
total_Si_peak_dt.to_csv('results/total_Si_peak_dt.csv', index=True)
first_Si_peak_dt.to_csv('results/first_Si_peak_dt.csv', index=True)
second_Si_peak_dt.to_csv('results/second_Si_peak_dt.csv', index=True)
total_Si_late_dt.to_csv('results/total_Si_late_dt.csv', index=True)
first_Si_late_dt.to_csv('results/first_Si_late_dt.csv', index=True)
second_Si_late_dt.to_csv('results/second_Si_late_dt.csv', index=True)

############################################################################################
############################################################################################
# diffusive controlled reaction 
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','D','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.05, 1], # lamb
               [0.05, 1], # alpha
               [0.05, 1]] # kd
}
times = np.linspace(0,3000000,1000)
L = 2
x = 2
ts = 5
v = 0.001
Co = 1

param_values = saltelli.sample(problem, 2**1)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])


Y_early_dr = np.zeros(param_values.shape[0])
Y_peak_dr = np.zeros(param_values.shape[0])
Y_late_dr = np.zeros(param_values.shape[0])

btc_data = []

for i, X in enumerate(param_values):
    Y_early_dr[i], Y_peak_dr[i], Y_late_dr[i], concentrations, adaptive_times = model.concentration_106_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    print(f'Diffusion reaction iteration: {i}')

    btc_data.append({
        "index": i,
        "params": X.tolist(),
        "times": adaptive_times,
        "concentrations": concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early_dr,
    'Peak': Y_peak_dr,
    'Late': Y_late_dr
})

metrics_df.to_csv('results/metrics_dr.csv', index=True)

with open('results/btc_data_dr.json', 'w') as f:
    json.dump(btc_data, f)

Si_early_dr = sobol.analyze(problem, Y_early_dr, print_to_console=False)
Si_peak_dr = sobol.analyze(problem, Y_peak_dr, print_to_console=False)
Si_late_dr = sobol.analyze(problem, Y_late_dr, print_to_console=False)

total_Si_early_dr, first_Si_early_dr, second_Si_early_dr = Si_early_dr.to_df()
total_Si_peak_dr, first_Si_peak_dr, second_Si_peak_dr = Si_peak_dr.to_df()
total_Si_late_dr, first_Si_late_dr, second_Si_late_dr = Si_late_dr.to_df()

total_Si_early_dr.to_csv('results/total_Si_early_dr.csv', index=True)
first_Si_early_dr.to_csv('results/first_Si_early_dr.csv', index=True)
second_Si_early_dr.to_csv('results/second_Si_early_dr.csv', index=True)
total_Si_peak_dr.to_csv('results/total_Si_peak_dr.csv', index=True)
first_Si_peak_dr.to_csv('results/first_Si_peak_dr.csv', index=True)
second_Si_peak_dr.to_csv('results/second_Si_peak_dr.csv', index=True)
total_Si_late_dr.to_csv('results/total_Si_late_dr.csv', index=True)
first_Si_late_dr.to_csv('results/first_Si_late_dr.csv', index=True)
second_Si_late_dr.to_csv('results/second_Si_late_dr.csv', index=True)

############################################################################################
############################################################################################
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
times = np.linspace(0,300,300)
L = 2
x = 2
ts = 5
v = 0.5
Co = 1

param_values = saltelli.sample(problem, 2**1)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])

Y_early_at = np.zeros(param_values.shape[0])
Y_peak_at = np.zeros(param_values.shape[0])
Y_late_at = np.zeros(param_values.shape[0])

btc_data = []

for i, X in enumerate(param_values):
    Y_early_at[i], Y_peak_at[i], Y_late_at[i], concentrations, adaptive_times = model.concentration_106_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    print(f'Advection transport iteration: {i}')


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

metrics_df.to_csv('results/metrics_at.csv', index=True)

with open('results/btc_data_at.json', 'w') as f:
    json.dump(btc_data, f)

Si_early_at = sobol.analyze(problem, Y_early_at, print_to_console=False)
Si_peak_at = sobol.analyze(problem, Y_peak_at, print_to_console=False)
Si_late_at = sobol.analyze(problem, Y_late_at, print_to_console=False)

total_Si_early_at, first_Si_early_at, second_Si_early_at = Si_early_at.to_df()
total_Si_peak_at, first_Si_peak_at, second_Si_peak_at = Si_peak_at.to_df()
total_Si_late_at, first_Si_late_at, second_Si_late_at = Si_late_at.to_df()

total_Si_early_at.to_csv('results/total_Si_early_at.csv', index=True)
first_Si_early_at.to_csv('results/first_Si_early_at.csv', index=True)
second_Si_early_at.to_csv('results/second_Si_early_at.csv', index=True)
total_Si_peak_at.to_csv('results/total_Si_peak_at.csv', index=True)
first_Si_peak_at.to_csv('results/first_Si_peak_at.csv', index=True)
second_Si_peak_at.to_csv('results/second_Si_peak_at.csv', index=True)
total_Si_late_at.to_csv('results/total_Si_late_at.csv', index=True)
first_Si_late_at.to_csv('results/first_Si_late_at.csv', index=True)
second_Si_late_at.to_csv('results/second_Si_late_at.csv', index=True)

############################################################################################
############################################################################################
#%%
# advective controlled reaction 
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','D','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.6, 1], # lamb
               [0.5, 1], # alpha
               [0.5, 1]] # kd
}

param_values = saltelli.sample(problem, 2**1)
times = np.linspace(0,4000,1000)
L = 2
x = 2
ts = 5
v = 0.5
Co = 1

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','lamb','alpha','kd'])


Y_early_ar = np.zeros(param_values.shape[0])
Y_peak_ar = np.zeros(param_values.shape[0])
Y_late_ar = np.zeros(param_values.shape[0])

btc_data = []

for i, X in enumerate(param_values):
    Y_early_ar[i], Y_peak_ar[i], Y_late_ar[i], concentrations, adaptive_times = model.concentration_106_new_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5], Co=Co, v=v, ts=ts, L=L, x=x)
    print(f'Advection reaction iteration: {i}')

    btc_data.append({
        "index": i,
        "params": X.tolist(),
        "times": adaptive_times,
        "concentrations": concentrations.tolist()
        })

metrics_df = pd.DataFrame({
    'Early': Y_early_ar,
    'Peak': Y_peak_ar,
    'Late': Y_late_ar
})

metrics_df.to_csv('results/metrics_ar.csv', index=True)

with open('results/btc_data_ar.json', 'w') as f:
    json.dump(btc_data, f)

Si_early_ar = sobol.analyze(problem, Y_early_ar, print_to_console=False)
Si_peak_ar = sobol.analyze(problem, Y_peak_ar, print_to_console=False)
Si_late_ar = sobol.analyze(problem, Y_late_ar, print_to_console=False)

total_Si_early_ar, first_Si_early_ar, second_Si_early_ar = Si_early_ar.to_df()
total_Si_peak_ar, first_Si_peak_ar, second_Si_peak_ar = Si_peak_ar.to_df()
total_Si_late_ar, first_Si_late_ar, second_Si_late_ar = Si_late_ar.to_df()

total_Si_early_ar.to_csv('results/total_Si_early_ar.csv', index=True)
first_Si_early_ar.to_csv('results/first_Si_early_ar.csv', index=True)
second_Si_early_ar.to_csv('results/second_Si_early_ar.csv', index=True)
total_Si_peak_ar.to_csv('results/total_Si_peak_ar.csv', index=True)
first_Si_peak_ar.to_csv('results/first_Si_peak_ar.csv', index=True)
second_Si_peak_ar.to_csv('results/second_Si_peak_ar.csv', index=True)
total_Si_late_ar.to_csv('results/total_Si_late_ar.csv', index=True)
first_Si_late_ar.to_csv('results/first_Si_late_ar.csv', index=True)
second_Si_late_ar.to_csv('results/second_Si_late_ar.csv', index=True)
