
import numpy as np
import pandas as pd
from scipy import special
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
#%%
# diffusive controlled transport 
problem = {
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 0.05], # v
               [0, 0.0005], # lamb
               [0, 0.0005], # alpha
               [0, 0.0005]] # kd 
}

times = np.linspace(0,12000,24000)
L = 2

param_values = saltelli.sample(problem, 2**4)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd'])

# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']
diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]

Y_early_dt = np.zeros(diff_trans.shape[0])
Y_peak_dt = np.zeros(diff_trans.shape[0])
Y_late_dt = np.zeros(diff_trans.shape[0])

for i, X in enumerate(param_values):
    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    print(f'Diffusive transport iteration: {i}')

Si_early_dt = sobol.analyze(problem, Y_early_dt, print_to_console=False)
Si_peak_dt = sobol.analyze(problem, Y_peak_dt, print_to_console=False)
Si_late_dt = sobol.analyze(problem, Y_late_dt, print_to_console=False)

total_Si_early_dt, first_Si_early_dt, second_Si_early_dt = Si_early_dt.to_df()
total_Si_peak_dt, first_Si_peak_dt, second_Si_peak_dt = Si_peak_dt.to_df()
total_Si_late_dt, first_Si_late_dt, second_Si_late_dt = Si_late_dt.to_df()

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
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 0.05], # v
               [0.05, 1], # lamb
               [0.05, 1], # alpha
               [0.05, 1]] # kd
}

param_values = saltelli.sample(problem, 2**4)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd'])

# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']
diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]

Y_early_dr = np.zeros(diff_reaction.shape[0])
Y_peak_dr = np.zeros(diff_reaction.shape[0])
Y_late_dr = np.zeros(diff_reaction.shape[0])

for i, X in enumerate(param_values):
    Y_early_dr[i], Y_peak_dr[i], Y_late_dr[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    print(f'Diffusion reaction iteration: {i}')

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
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.1, 1], # v
               [0, 0.05], # lamb
               [0, 0.05], # alpha
               [0, 0.05]] # kd
}

param_values = saltelli.sample(problem, 2**4)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd'])

# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']
adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]

Y_early_at = np.zeros(adv_trans.shape[0])
Y_peak_at = np.zeros(adv_trans.shape[0])
Y_late_at = np.zeros(adv_trans.shape[0])

for i, X in enumerate(param_values):
    Y_early_at[i], Y_peak_at[i], Y_late_at[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    print(f'Advection transport iteration: {i}')

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
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.05, 1], # v
               [0.6, 1], # lamb
               [0.5, 1], # alpha
               [0.5, 1]] # kd
}

param_values = saltelli.sample(problem, 2**4)
times = np.linspace(0,12000,24000)
L = 2
params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']
adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]

Y_early_ar = np.zeros(adv_reaction.shape[0])
Y_peak_ar = np.zeros(adv_reaction.shape[0])
Y_late_ar = np.zeros(adv_reaction.shape[0])

for i, X in enumerate(param_values):
    Y_early_ar[i], Y_peak_ar[i], Y_late_ar[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    print(f'Advection reaction iteration: {i}')

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
