#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:30:33 2024

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
import model
from SALib.sample import saltelli
import seaborn as sns
import matplotlib.patches as mpatches
from pandas.plotting import table
#%%
# create a group of parameters to sample from
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.001, 2], # D
               [0.001, 1], # v
               [0, 0.5], # lamb
               [0, 1], # alpha
               [0, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**8)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

# plotting
plt.figure(figsize=(10,8))
plt.style.use('ggplot')  # Applying ggplot style

plt.subplot(2,2,1)
sns.kdeplot(params_df['Pe'], c='black')
plt.axvline(1, c = 'red')
plt.xscale('log')
plt.title('Sampled Peclet #')

plt.subplot(2,2,2)
sns.kdeplot(params_df['Da'], c='black')
plt.axvline(1, c = 'red')
plt.xscale('log')
plt.title('Sampled Damkohler #')


plt.subplot(2,2,3)
peclet_counts = params_df['Pe'].apply(lambda x: 'Dispersion Dominated' if x < 1 else 'Advection Dominated').value_counts()
peclet_counts = peclet_counts.reindex(['Dispersion Dominated', 'Advection Dominated'])
plt.bar(peclet_counts.index, peclet_counts.values, color=['blue','red'])
plt.title('System count by Peclet #')

plt.subplot(2,2,4)
dahmkoler_counts = params_df['Da'].apply(lambda x: 'Transport Dominated' if x < 1 else 'Reaction Dominated').value_counts()
dahmkoler_counts = dahmkoler_counts.reindex(['Transport Dominated', 'Reaction Dominated'])
plt.bar(dahmkoler_counts.index, dahmkoler_counts.values, color=['orange','purple'])
plt.title('System count by Damkohler #')

plt.suptitle('Parameter Ranges', fontsize=18)
plt.tight_layout()

# column_labels = ['Parameter', 'Minimum', 'Maximum']
# column_widths = [0.5,0.2,0.2]
# cell_text = [
#     ['Theta', str(params_df['theta'].min()), str(params_df['theta'].max())],
#     ['Bulk Density', str(params_df['rho_b'].min()), str(params_df['rho_b'].max())],
#     ['Dispersion', str(params_df['D'].min()), str(params_df['D'].max())],
#     ['Pore Velocity', str(params_df['v'].min()), str(params_df['v'].max())],
#     ['Lambda', str(params_df['lamb'].min()), str(params_df['lamb'].max())],
#     ['Alpha', str(params_df['alpha'].min()), str(params_df['alpha'].max())],
#     ['Sorption coefficient', str(params_df['kd'].min()), str(params_df['kd'].max())],
#     ['Initial Concentration', str(params_df['Co'].min()), str(params_df['Co'].max())]
# ]

# plt.table(cellText=cell_text,
#           colLabels=column_labels,
#           colWidths=column_widths,
#           cellLoc='center',
#           loc='center right',
#           bbox=[1.1,0,0.5,2.5])
# #table.auto_set_font_size(False)
# #table.set_fontsize(8)
# #table.scale(1.2, 1.2)
plt.show()



print('Peclet range: ' +str(params_df['Pe'].min()) +' to ' + str(params_df['Pe'].max()))
print('Dahmkohler range: ' +str(params_df['Da'].min()) +' to ' + str(params_df['Da'].max()))

#%% scatter plot version
# subset by Pe and Da
diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]
adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]
diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]
adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]

# plotting
plt.figure(figsize=(8,8))
plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# labeling
plt.annotate('Diffusive controlled reaction', (10**-1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled reaction', (10**1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Diffusive controlled transport', (10**-1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled transport', (10**1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')

# formatting
plt.xlabel('Peclet Number')
plt.ylabel('Damkohler Number')
plt.xscale('log')
plt.yscale('log')
plt.axhline(1, c='black', linewidth=2)
plt.axvline(1, c='black', linewidth=2)
plt.show()


#%% create BTCs from sampled parameters

# times = np.linspace(1, 50, 50)
# btc_data = []
# legend_labels_pe = []
# legend_labels_da = []

# for i in range(len(params_df)):
#     theta = params_df.iloc[i, 0]
#     rho_b = params_df.iloc[i, 1]
#     D = params_df.iloc[i, 2]
#     v = params_df.iloc[i, 3]
#     lamb = params_df.iloc[i, 4]
#     alpha = params_df.iloc[i, 5]
#     kd = params_df.iloc[i, 6]
#     Co = params_df.iloc[i, 7]

#     # Calculate Peclet number and Dahmkoler number
#     Pe = v * 2 / D
#     if Pe > 1:
#         Da = lamb * 2 / v
#     else:
#         Da = lamb * 2**2 / D
#     # Generate BTC
#     concs = np.array([model.concentration_102(times, theta, rho_b, D, v, lamb, alpha, kd, Co)])
    
#     # Append concentration data to btc_data
#     btc_data.append(concs)

#     # Store legend labels
#     if Pe > 1 and 'Advection Dominated' not in legend_labels_pe:
#         legend_labels_pe.append('Advection Dominated')
#     elif Pe <= 1 and 'Dispersion Dominated' not in legend_labels_pe:
#         legend_labels_pe.append('Dispersion Dominated')


#     if Da > 1 and 'Reaction Dominated' not in legend_labels_da:
#         legend_labels_da.append('Reaction Dominated')
#     elif Da <= 1 and 'Transport Dominated' not in legend_labels_da:
#         legend_labels_da.append('Transport Dominated')
        
# # Convert btc_data to numpy array and flatten
# btc_data = np.array(btc_data).reshape(len(params_df), -1)

# # Plotting BTCs
# plt.figure(figsize=(9,5))
# plt.subplot(1,2,1)
# for i, concs in enumerate(btc_data):
#     # Assign color based on Peclet number
#     Pe = params_df.iloc[i, 3] * 2 / params_df.iloc[i, 2]
#     color = 'red' if Pe > 1 else 'blue'

#     # Plot BTC
#     plt.plot(times, concs, alpha=0.3, color=color)

# legend_handles_pe = [mpatches.Patch(color='red', alpha=0.3), mpatches.Patch(color='blue', alpha=0.3)]
# plt.legend(legend_handles_pe, ['Advection Dominated', 'Dispersion Dominated'])
# plt.xlabel('Time (minutes)')
# plt.ylabel('Concentration (mg/m)')
# plt.title('Peclet #')

# plt.subplot(1,2,2)

# for i, concs in enumerate(btc_data):
#     # Assign color based on Peclet number
#     Pe = params_df.iloc[i, 3] * 2 / params_df.iloc[i, 2]
#     if Pe > 1:
#         Da = params_df.iloc[i,4] * 2 / params_df.iloc[i,3]
#     else:
#         Da = params_df.iloc[i,4] * 2**2 / params_df.iloc[i,2]
#     color = 'purple' if Da > 1 else 'orange'

#     # Plot BTC
#     plt.plot(times, concs, alpha=0.3, color=color)

# legend_handles_da = [mpatches.Patch(color='purple', alpha=0.3), mpatches.Patch(color='orange', alpha=0.3)]
# plt.legend(legend_handles_da, ['Reaction Dominated', 'Transport Dominated'])

# # Adding labels and title
# plt.xlabel('Time (minutes)')
# plt.ylabel('Concentration (mg/m)')
# plt.title('Dahmkoler #', fontsize=12)

# plt.subplots_adjust(top=0.85)
# #plt.tight_layout()
# plt.suptitle('Contaminant Breakthrough Curves', fontsize=18)

# # Displaying the plot
# plt.show()

#%%

# times = np.linspace(1,80,80)

# theta = 0.25
# rho_b = 1.5
# D = 0.05
# v = 0.01
# lamb = 0
# alpha = 1
# kd = 0
# Co = 1
# ts = 5
# L = 2
# x = 2

# print(v*2/D)

# concs_102 = model.concentration_102(times,theta,rho_b,D,v,lamb,alpha,kd,Co) # generates a list of mpf objects
# concs_102_float = [float(conc) for conc in concs_102] # convert to floats
# concs_102_array = np.array(concs_102_float) # convert to array

# plt.plot(times, concs_102_array, label = 'dispersion dominated')

# theta = 0.25
# rho_b = 1.5
# D = 0.01
# v = 1
# lamb = 0
# alpha = 0
# kd = 0
# Co = 1
# ts = 5
# L = 2
# x = 2

# print(v*2/D)

# concs_102 = model.concentration_102(times,theta,rho_b,D,v,lamb,alpha,kd,Co) # generates a list of mpf objects
# concs_102_float = [float(conc) for conc in concs_102] # convert to floats
# concs_102_array = np.array(concs_102_float) # convert to array
# plt.plot(times, concs_102_array, label = 'advection dominated')

# plt.ylim(0,1)
# plt.legend()
# plt.show()
#%%
# diffusive controlled transport 
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 0.05], # v
               [0, 0.0005], # lamb
               [0, 0.0005], # alpha
               [0, 0.0005], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**8)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

# scatter plot version
# subset by Pe and Da
# diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]
# adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]
diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]
# adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]

# # plotting
# plt.figure(figsize=(8,8))
# plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
# plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
# plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
# plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# # labeling
# plt.annotate('Diffusive controlled reaction', (10**-1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled reaction', (10**1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Diffusive controlled transport', (10**-1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled transport', (10**1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')

# # formatting
# plt.xlabel('Peclet Number')
# plt.ylabel('Damkohler Number')
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(1, c='black', linewidth=2)
# plt.axvline(1, c='black', linewidth=2)
# plt.show()

#%%
# diffusive controlled reaction 
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 0.05], # v
               [0.05, 1], # lamb
               [0.05, 1], # alpha
               [0.05, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**8)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

# scatter plot version
# subset by Pe and Da
diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]
# adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]
# diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]
# adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]

# # plotting
# plt.figure(figsize=(8,8))
# plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
# plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
# plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
# plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# # labeling
# plt.annotate('Diffusive controlled reaction', (10**-1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled reaction', (10**1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Diffusive controlled transport', (10**-1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled transport', (10**1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')

# # formatting
# plt.xlabel('Peclet Number')
# plt.ylabel('Damkohler Number')
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(1, c='black', linewidth=2)
# plt.axvline(1, c='black', linewidth=2)
# plt.show()

#%%
# advective controlled transport 
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.1, 1], # v
               [0, 0.05], # lamb
               [0, 0.05], # alpha
               [0, 0.05], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**8)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

# scatter plot version
# subset by Pe and Da
# diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]
adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]
# diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]
# adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]

# # plotting
# plt.figure(figsize=(8,8))
# plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
# plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
# plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
# plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# # labeling
# plt.annotate('Diffusive controlled reaction', (10**-1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled reaction', (10**1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Diffusive controlled transport', (10**-1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled transport', (10**1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')

# # formatting
# plt.xlabel('Peclet Number')
# plt.ylabel('Damkohler Number')
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(1, c='black', linewidth=2)
# plt.axvline(1, c='black', linewidth=2)
# plt.show()

#%%
# advective controlled reaction 
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.05, 1], # v
               [0.6, 1], # lamb
               [0.5, 1], # alpha
               [0.5, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**8)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

# scatter plot version
# subset by Pe and Da
#diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]
#adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]
#diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]
adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]

# plotting
# plt.figure(figsize=(8,8))
# plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
# plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
# plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
# plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# # labeling
# plt.annotate('Diffusive controlled reaction', (10**-1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled reaction', (10**1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Diffusive controlled transport', (10**-1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')
# plt.annotate('Advective controlled transport', (10**1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')

# # formatting
# plt.xlabel('Peclet Number')
# plt.ylabel('Damkohler Number')
# plt.xscale('log')
# plt.yscale('log')
# plt.axhline(1, c='black', linewidth=2)
# plt.axvline(1, c='black', linewidth=2)
# plt.show()

#%%

# plotting
plt.figure(figsize=(8,8))
plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# labeling
plt.annotate('Diffusive controlled reaction', (10**-1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled reaction', (10**1.4, 10**2.4), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Diffusive controlled transport', (10**-1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled transport', (10**1.4, 10**-2.4), fontweight='bold', fontsize=10, ha='center')

# formatting
plt.xlabel('Peclet Number')
plt.ylabel('Damkohler Number')
plt.xscale('log')
plt.yscale('log')
plt.axhline(1, c='black', linewidth=2)
plt.axvline(1, c='black', linewidth=2)
plt.show()

#%% new version
# since dispersion is dependent on velocity and diffusion, I need to redo this figure
# diffusion is now variable (range 8.64e-7 to 8.64e-5 (m^2/day))

problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [0.001, 10], # dispersivity
               [0.00001, 1], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])

params_df['v'] = 0.5

# calculate dispersion
params_df['D'] = params_df['v'] * params_df['dispersivity']

# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

adv_reaction = params_df[(params_df['Pe'] > 1) & (params_df['Da'] > 1)]
adv_trans = params_df[(params_df['Pe'] > 1) & (params_df['Da'] < 1)]
diff_reaction = params_df[(params_df['Pe'] < 1) & (params_df['Da'] > 1)]
diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]



plt.figure(figsize=(8,8))
plt.scatter(diff_reaction['Pe'], diff_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Diffusive controlled reaction')
plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
plt.scatter(diff_trans['Pe'], diff_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# labeling
plt.annotate('Diffusive controlled reaction', (10**-1.5, 10), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled reaction', (10**1.5, 10), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Diffusive controlled transport', (10**-1.5, 10**-2), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled transport', (10**1.5, 10**-2), fontweight='bold', fontsize=10, ha='center')

# formatting
plt.xlabel('Peclet Number')
plt.ylabel('Damkohler Number')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-3,1e3)
plt.ylim(1e-3,1e3)
plt.axhline(1, c='black', linewidth=2)
plt.axvline(1, c='black', linewidth=2)
plt.show()

#%%
# adv reaction
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(0.001), np.log10(1.8)], # dispersivity
               [np.log10(0.28), np.log10(10)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

adv_reaction = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
adv_reaction['dispersivity'] = 10**adv_reaction['dispersivity']
adv_reaction['lamb'] = 10**adv_reaction['lamb']

adv_reaction['v'] = 0.5
# calculate dispersion
adv_reaction['D'] = adv_reaction['v'] * adv_reaction['dispersivity']
# calculate Peclet and Dahmkoler
adv_reaction['Pe'] = (adv_reaction['v'] * 2) / adv_reaction['D']
adv_reaction['Da'] = (adv_reaction['lamb'] * 2) / adv_reaction['v']

# advective transport
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(0.001), np.log10(1.8)], # dispersivity
               [np.log10(0.00001), np.log10(0.25)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

adv_trans = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
adv_trans['dispersivity'] = 10**adv_trans['dispersivity']
adv_trans['lamb'] = 10**adv_trans['lamb']

adv_trans['v'] = 0.5
# calculate dispersion
adv_trans['D'] = adv_trans['v'] * adv_trans['dispersivity']
# calculate Peclet and Dahmkoler
adv_trans['Pe'] = (adv_trans['v'] * 2) / adv_trans['D']
adv_trans['Da'] = (adv_trans['lamb'] * 2) / adv_trans['v']

# dispersive reaction
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(2), np.log10(100)], # dispersivity
               [np.log10(0.28), np.log10(10)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

disp_reaction = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
disp_reaction['dispersivity'] = 10**disp_reaction['dispersivity']
disp_reaction['lamb'] = 10**disp_reaction['lamb']

disp_reaction['v'] = 0.5
# calculate dispersion
disp_reaction['D'] = disp_reaction['v'] * disp_reaction['dispersivity']
# calculate Peclet and Dahmkoler
disp_reaction['Pe'] = (disp_reaction['v'] * 2) / disp_reaction['D']
disp_reaction['Da'] = (disp_reaction['lamb'] * 2) / disp_reaction['v']


# dispersive transport
problem = {
    'num_vars': 6,
    'names': ['theta', 'rho_b','dispersivity','lamb','alpha','kd'],
    'bounds': [[0.25, 0.7], # theta - porosity
               [0.29, 1.74], # rho_b - bulk density
               [np.log10(2), np.log10(100)], # dispersivity
               [np.log10(0.00001), np.log10(0.25)], # lamb - first order decay rate constant
               [0, 24], # alpha - first order desorption rate constant
               [0.01, 100]] # kd - sorption distribution coefficient
}

param_values = saltelli.sample(problem, 2**8)

disp_trans = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','dispersivity','lamb','alpha','kd'])
# convert back to real space
disp_trans['dispersivity'] = 10**disp_trans['dispersivity']
disp_trans['lamb'] = 10**disp_trans['lamb']
disp_trans['v'] = 0.5
# calculate dispersion
disp_trans['D'] = disp_trans['v'] * disp_trans['dispersivity']
# calculate Peclet and Dahmkoler
disp_trans['Pe'] = (disp_trans['v'] * 2) / disp_trans['D']
disp_trans['Da'] = (disp_trans['lamb'] * 2) / disp_trans['v']

plt.figure(figsize=(8,8))
plt.scatter(disp_reaction['Pe'], disp_reaction['Da'], c='blue', alpha=0.7, s = 5, label='Dispersion controlled reaction')
plt.scatter(adv_reaction['Pe'], adv_reaction['Da'], c='purple', alpha=0.7, s=5, label='Advective controlled reaction')
plt.scatter(disp_trans['Pe'], disp_trans['Da'], c='orange', alpha=0.7, s=5, label='Diffusive controlled transport')
plt.scatter(adv_trans['Pe'], adv_trans['Da'], c='red', alpha=0.7, s=5, label='Advective controlled transport')

# labeling
plt.annotate('Dispersion controlled reaction', (10**-1.5, 100), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled reaction', (10**1.5, 100), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Dispersion controlled transport', (10**-1.5, 10**-2), fontweight='bold', fontsize=10, ha='center')
plt.annotate('Advective controlled transport', (10**1.5, 10**-2), fontweight='bold', fontsize=10, ha='center')

# formatting
plt.xlabel('Peclet Number')
plt.ylabel('Damkohler Number')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-3,1e3)
plt.ylim(1e-3,1e3)
plt.axhline(1, c='black', linewidth=2)
plt.axvline(1, c='black', linewidth=2)
plt.show()




